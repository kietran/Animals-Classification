import torch
from torch import nn
from torch import optim
from datasets import AnimalDataset
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from tqdm.autonotebook import tqdm
import argparse
import os
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", "-d", type=str, default='Dataset/animals')
    parser.add_argument("--checkpoint_dir", "-c", type=str, default='trained_models')
    parser.add_argument("--checkpoint", "-p", type=str, default=None)
    parser.add_argument("--tensorboard_dir", "-t", type=str, default='tensorboard/animals')
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--learning_rate", "-l", type=float, default=0.01)
    parser.add_argument("--img_size", "-i", type=int, default=224)
    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: 
    plt.imshow(cm, interpolation='nearest', cmap='Wistia')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    writer.add_figure('confusion_matrix', figure, epoch)

def train(args):
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if not os.path.isdir(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    writer = SummaryWriter(args.tensorboard_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        ToTensor(),
        Resize((args.img_size, args.img_size))
    ])

    train_dataset = AnimalDataset(root_path=args.dataset_dir, is_train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    valid_dataset = AnimalDataset(root_path=args.dataset_dir, is_train=False, transform=transform)
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    num_iters = len(train_dataloader)
    # model = AdvancedCNN(num_classes=len(train_dataset.categories))
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=10, bias=True)
    optimizer = optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=0.9)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Load checkpoint if requested
    if args.checkpoint and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        start_epoch =  checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model_params"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        best_acc = -1
        start_epoch = 0

    best_acc = -1
    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        all_losses = []
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)

            # Forward
            loss = criterion(output, labels)
            all_losses.append(loss.item())
            loss_val = np.mean(all_losses)
            progress_bar.set_description(f"(Train) Epoch: {epoch+1}/{args.epochs}. Loss: {loss_val}")
            writer.add_scalar('Train/Loss', loss_val, epoch*num_iters+iter)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        all_labels = []
        all_predictions = []
        all_losses = []
        for iter, (images, labels) in enumerate(valid_dataloader):
            with torch.inference_mode():
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                all_losses.append(loss.item())
                all_labels.extend(labels.tolist())
                all_predictions.extend(torch.argmax(output, dim=1).tolist())

        acc_score = accuracy_score(all_labels, all_predictions)
        loss_val = np.mean(all_losses)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        plot_confusion_matrix(writer, conf_matrix, train_dataset.categories, epoch)
        print(f"(Valid) Epoch: {epoch+1}/{args.epochs}. Accuracy: {acc_score}. Loss: {loss_val}")
        writer.add_scalar('Val/Loss', loss_val, epoch)
        writer.add_scalar('Val/Acc', acc_score, epoch)

        # Save checkpoint
        checkpoint = {
            "epoch": epoch+1, 
            "model_params": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pt"))
        if acc_score > best_acc:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best.pt"))
        
 
if __name__ == "__main__":
    args = add_args()
    train(args)