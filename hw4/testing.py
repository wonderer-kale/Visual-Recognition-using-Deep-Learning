import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from train import PromptIRModel


def test(model, test_dir, output_dir, transform, device):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    test_img_dir = os.path.join(test_dir, "degraded")
    test_img_filenames = sorted(os.listdir(test_img_dir))

    with torch.no_grad():
        for filename in tqdm(test_img_filenames, desc="Testing"):
            img_path = os.path.join(test_img_dir, filename)
            img = Image.open(img_path).convert('RGB')

            if transform:
                img_tensor = transform(img).unsqueeze(0).to(device)

            # Forward pass
            output = model(img_tensor)

            # Convert output tensor to image
            output_img = output.squeeze(0).cpu().clamp(0, 1)
            output_img = output_img.permute(1, 2, 0).numpy()
            output_img = (output_img * 255).astype(np.uint8)
            output_img = Image.fromarray(output_img)

            # Save the restored image
            output_path = os.path.join(output_dir, filename)
            output_img.save(output_path)


def main():
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    ckpt_path = "train_ckpt/best-epoch=epoch=265-val_psnr=val_psnr=28.99.ckpt"
    model_module = PromptIRModel.load_from_checkpoint(
        ckpt_path, map_location=device)
    model = model_module.net.to(device)  # Extract the actual PromptIR network

    print("Loaded best model from checkpoint")

    # Run inference
    test(model, "../hw4_dataset/test", "restored_images", transform, device)
    print("Testing completed. Restored images saved to ./restored_images")

    # Save restored results to NPZ
    folder_path = 'restored_images'
    output_npz = 'pred.npz'
    images_dict = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            image = Image.open(file_path).convert('RGB')
            img_array = np.array(image)
            img_array = np.transpose(img_array, (2, 0, 1))
            images_dict[filename] = img_array

    np.savez(output_npz, **images_dict)
    print(f"Saved {len(images_dict)} images to {output_npz}")


if __name__ == "__main__":
    main()
