def generate(prompt, steps=100, cfg_scale=10.0, resolution=1024):
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
  
    text_emb = text_encoder(input_ids)
    
    
    latents = torch.randn(1, 4, resolution//8, resolution//8, device="cuda")
    
    
    uncond_emb = text_encoder(tokenizer("", return_tensors="pt").input_ids.to("cuda"))
    text_emb = torch.cat([uncond_emb, text_emb])
    
  
    for t in reversed(range(steps)):
        
        latent_model_input = torch.cat([latents] * 2)
        
        
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, text_emb)
        
      
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        
        
        latents = noise_scheduler.step(noise_pred, t, latents)
    
    
    image = vae.decode(latents / 0.18215).sample
    return image
