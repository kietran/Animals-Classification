from torch import nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.Dropout()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.Dropout()
        )

        self.fc3 = nn.Linear(in_features=128, out_features=10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.make_block(in_channels=3, out_channels=16)
        self.conv2 = self.make_block(in_channels=16, out_channels=32)
        self.conv3 = self.make_block(in_channels=32, out_channels=64)
        self.conv4 = self.make_block(in_channels=64, out_channels=128)
        self.conv5 = self.make_block(in_channels=128, out_channels=128)

        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=6272, out_features=2048),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # model = SimpleCNN()
    # fake_cifar_dataset = torch.rand(8, 3, 32, 32)
    # output = model(fake_cifar_dataset)
    # print(output.shape)

    model = AdvancedCNN()
    fake_animal_dataset = torch.rand(8, 3, 224, 224)
    output = model(fake_animal_dataset)
    print(output.shape)