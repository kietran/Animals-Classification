from torch.utils.data.dataset import Dataset
from PIL import Image
import os

class AnimalDataset(Dataset):
    def __init__(self, root_path, is_train, transform):
        if (is_train):
            data_path = os.path.join(root_path, 'train')
        else:
            data_path = os.path.join(root_path, 'test')
        
        self.categories = []
        for dir in os.listdir(data_path):
            self.categories.append(dir)
        sub_dirs = [os.path.join(data_path, dir) for dir in os.listdir(data_path)]
        
        self.labels = []
        self.image_paths = []
        for sub_dir in sub_dirs:
            for file_path in os.listdir(sub_dir):
                self.labels.append(self.categories.index(os.path.basename(sub_dir)))
                self.image_paths.append(os.path.join(sub_dir, file_path))

        self.transform = transform
                
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.image_paths[index]

        image = Image.open(image).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label