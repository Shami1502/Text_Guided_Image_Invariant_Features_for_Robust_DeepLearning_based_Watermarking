import numpy as np
from PIL import Image
import io
from torchvision import transforms

class AddNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        np_img = np.array(img)
        noise = np.random.normal(self.mean, self.std, np_img.shape)
        noisy_img = np_img + noise
        noisy_img = np.clip(noisy_img, 0, 255)
        return Image.fromarray(noisy_img.astype(np.uint8))

class AddJPEGArtifacts(object):
    def __init__(self, quality=50):
        self.quality = quality

    def __call__(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)

augmentation = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomApply([
        transforms.RandomOrder([
            AddNoise(mean=0, std=5),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            AddJPEGArtifacts(quality=50)
        ])
    ], p=0.5),
    transforms.RandomApply([
        AddNoise(mean=0, std=25)
    ], p=0.3)
])
