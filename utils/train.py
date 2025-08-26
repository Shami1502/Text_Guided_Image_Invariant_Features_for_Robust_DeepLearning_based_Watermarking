import os
import torch
from tqdm import tqdm
import clip
from .losses import contrastive_loss
from .negatives import get_hard_negative_indices

def train_clip_invariant(model, dataloader, optimizer, device, num_epochs=50, save_interval=10):
    model.train()
    save_dir = './clip_model_F'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            images, captions, distorted_images = batch
            images = images.to(device)
            distorted_images = distorted_images.to(device)
            captions = clip.tokenize(captions).to(device)

            image_proj, distorted_proj, text_proj = model(images, distorted_images, captions)

            hard_neg_indices = get_hard_negative_indices(image_proj, threshold=0.5)
            negative_images = images[hard_neg_indices]
            negative_distorted_images = distorted_images[hard_neg_indices]

            negative_image_proj, negative_distorted_proj, _ = model(negative_images, negative_distorted_images, captions)

            loss = contrastive_loss(image_proj, distorted_proj, text_proj,
                                    negative_image_proj, negative_distorted_proj)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}")

        if (epoch + 1) % save_interval == 0:
            path = os.path.join(save_dir, f'model_4096__F_{epoch+1}.pt')
            torch.save(model.state_dict(), path)
            print(f"Model for epoch {epoch+1} saved to {path}")
