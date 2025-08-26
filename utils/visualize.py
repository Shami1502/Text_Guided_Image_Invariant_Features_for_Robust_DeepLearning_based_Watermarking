import matplotlib.pyplot as plt

def visualize_samples(image, distorted_image, caption):
    image = image.permute(1, 2, 0).cpu().numpy()
    distorted_image = distorted_image.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title(f"Original Image\nCaption: {caption}")
    ax[0].axis('off')

    ax[1].imshow(distorted_image)
    ax[1].set_title("Distorted Image")
    ax[1].axis('off')

    plt.show()
