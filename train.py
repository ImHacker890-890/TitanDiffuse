import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# 1. Dataset Class
class TextImageDataset(Dataset):
    def __init__(self, root_dir=".train", size=512):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Get corresponding prompt file
        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(self.root_dir, f"{base_name}_prompt.txt")
        
        if not os.path.exists(txt_path):
            txt_path = os.path.join(self.root_dir, f"{base_name}.txt")
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        
        return {"pixel_values": image, "text": prompt}

# 2. Training Setup
def main():
    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=2,
        log_with="tensorboard",
        project_dir="logs"
    )
    
    # Models
    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="unet")
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )
    
    # Dataset and DataLoader
    dataset = TextImageDataset(root_dir=".train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Optimizer
    optimizer = AdamW(unet.parameters(), lr=1e-5)
    
    # Prepare with Accelerator
    unet, text_encoder, optimizer, dataloader = accelerator.prepare(
        unet, text_encoder, optimizer, dataloader
    )
    
    # Training Loop
    num_epochs = 100
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = torch.randn_like(batch["pixel_values"])  # Simplified for example
                
                # Tokenize text
                inputs = tokenizer(
                    batch["text"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(accelerator.device)
                
                # Get text embeddings
                text_embeddings = text_encoder(inputs.input_ids)[0]
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps,
                    (latents.shape[0],),
                    device=accelerator.device
                ).long()
                
                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise)
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                progress_bar.set_postfix({"loss": loss.item()})
        
        # Save checkpoint
        if epoch % 5 == 0:
            accelerator.save_state(output_dir=f"checkpoints/epoch_{epoch}")

if __name__ == "__main__":
    main()
