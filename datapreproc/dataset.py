import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from .transforms import augmentation

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, text_file, transform=None, distortions=None):
        self.image_dir = image_dir
        self.transform = transform
        self.distortions = distortions if distortions else []
        self.image_caption_pairs = []

        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                image_name, caption = line.strip().split(',', 1)
                self.image_caption_pairs.append((image_name, caption))

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, index):
        image_name, caption = self.image_caption_pairs[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        width, height = image.size
        aspect_ratio = width / height
        new_width = 224
        new_height = int(new_width / aspect_ratio)
        image = image.resize((new_width, new_height))
        max_height = 224
        padding = (0, max_height - new_height, 0, 0)
        image = ImageOps.expand(image, padding)

        distorted_image = augmentation(image)

        if self.transform:
            image = self.transform(image)
            distorted_image = self.transform(distorted_image)

        return image, caption, distorted_image
