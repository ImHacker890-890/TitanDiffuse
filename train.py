from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

def train():
    
    accelerator = Accelerator(
        mixed_precision="bf16",
        fsdp_plugin=FSDPPlugin(
            sharding_strategy="FULL_SHARD",
            auto_wrap_policy=size_based_auto_wrap_policy(min_num_params=1e8),
            cpu_offload=True,
        )
    )
    
    
    text_encoder = MegaTextEncoder()
    unet = UltraUNet3D()
    
  
    optimizer = bitsandbytes.Adam8bit(
        list(unet.parameters()) + list(text_encoder.parameters()),
        lr=1e-5,
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )
    
    
    unet = accelerator.prepare_model(unet)
    text_encoder = accelerator.prepare_model(text_encoder)
    optimizer = accelerator.prepare_optimizer(optimizer)
    
    # Gradient checkpointing
    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable()
    
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    dataloader = accelerator.prepare(dataloader)
    
    
    for epoch in range(100):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Forward pass
                images = batch["images"].to(accelerator.device)
                text = batch["text"]
                
                
                input_ids = tokenizer(text, return_tensors="pt").input_ids.to(accelerator.device)
                text_emb = text_encoder(input_ids)
                
                
                noise = torch.randn_like(images)
                timesteps = torch.randint(0, 1000, (images.shape[0],)).to(accelerator.device)
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
                
                
                pred_noise = unet(noisy_images, timesteps, text_emb)
                
                
                loss = F.mse_loss(pred_noise, noise)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
