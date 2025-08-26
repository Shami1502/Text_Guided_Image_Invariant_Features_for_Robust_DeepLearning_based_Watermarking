import torch
import clip
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from dataprepoc.dataset import Flickr8kDataset
from models.clip_invariant import CLIPInvariantModel
from utils.train import train_clip_invariant

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.float()

if __name__ == "__main__":
    dataset = Flickr8kDataset(
        image_dir='Images',
        text_file='captions.txt',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])
    )

    dataset_length = len(dataset)
    train_size = int(0.9 * dataset_length)
    test_size = dataset_length - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    projection_dim = 4096
    model = CLIPInvariantModel(clip_model, projection_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_clip_invariant(model, train_loader, optimizer, device, num_epochs=100)
