from torchvision.datasets import VisionDataset
from PIL import Image
import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        self.images_paths = []
        self.labels = []

        self.categories = os.listdir(self.root)
        self.categories.remove("BACKGROUND_Google")

        for path in open(f"./Caltech101/{self.split}.txt"):
          path = path.replace("\n", "")
          category = path.split("/")[0]
          if(category != "BACKGROUND_Google"):
            image_path = self.root + path
            self.images_paths.append(image_path)
            self.labels.append(self.categories.index(category))

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        image, label = pil_loader(image_path), label[index]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        length = len(self.images_paths) # Provide a way to get the length (number of elements) of the dataset
        return length
