import torch
import cv2
import argparse
import numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', '-v', type=str, default=None, required=True)
    parser.add_argument('--image_size', '-s', type=int, default=224)
    parser.add_argument('--checkpoint', '-c', type=str, default='trained_models/best.pt')    
    args = parser.parse_args()
    return args

def inference_video(args):
    categories = ['spider', 'horse', 'butterfly', 'dog', 'chicken', 'elephant', 'sheep', 'cow', 'squirrel', 'cat']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = efficientnet_b0(EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=10, bias=True)
    model.to(device)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print('No checkpoint found')
        exit(0)

    cap = cv2.VideoCapture(args.video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))

    model.eval()
    softmax = nn.Softmax()
    with torch.inference_mode():
        while cap.isOpened():
            flag, frame = cap.read()
            if not flag:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (args.image_size, args.image_size))
            image = np.transpose(image, (2, 0, 1))/255.0
            image = torch.from_numpy(image).float()
            image = image[np.newaxis, :]
            image = image.to(device)

            prediction = model(image)
            proba = softmax(prediction)
            conf_score, max_index = torch.max(proba, dim=1)
            cv2.putText(frame, f"{categories[max_index]}({conf_score.item():0.2f})", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
            out.write(frame)

        out.release()
        cap.release()


if __name__ == "__main__":
    args = get_args()
    inference_video(args)