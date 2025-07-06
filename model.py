class MegaTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
      
        self.clip = CLIPTextModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s34B-b88K")
        
      
        self.context_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2048,
                nhead=32,
                dim_feedforward=8192,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=12,
        )
        
        
        self.proj = nn.Linear(2048, 4096)

    def forward(self, input_ids):
        
        clip_emb = self.clip(input_ids).last_hidden_state  # [B, L, 2048]
        
        
        context_emb = self.context_transformer(clip_emb)  # [B, L, 2048]
        
        # Конкатенация и проекция
        fused_emb = torch.cat([clip_emb, context_emb], dim=-1)
        return self.proj(fused_emb)  # [B, L, 4096]
