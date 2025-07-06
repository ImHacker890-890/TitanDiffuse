class UltraUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)
        
        # Downsample blocks 
        self.down_blocks = nn.ModuleList([
            DownBlock3D(320, 640, num_layers=3, attention=True),
            DownBlock3D(640, 1280, num_layers=3, attention=True),
            DownBlock3D(1280, 1920, num_layers=4, attention=True),
            DownBlock3D(1920, 2560, num_layers=4, attention=False),
        ])
        
        # Middle block (3D attention)
        self.mid_block = MidBlock3D(2560, num_heads=32)
        
        # Upsample blocks
        self.up_blocks = nn.ModuleList([
            UpBlock3D(2560, 1920, num_layers=4, attention=True),
            UpBlock3D(1920, 1280, num_layers=4, attention=True),
            UpBlock3D(1280, 640, num_layers=3, attention=True),
            UpBlock3D(640, 320, num_layers=3, attention=False),
        ])
        
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)
    
    def forward(self, x, t, text_emb):
        # x: [B, 4, H, W], t: [B], text_emb: [B, L, D]
        h = self.conv_in(x)
        
        # Downsample path
        skips = []
        for block in self.down_blocks:
            h = block(h, t, text_emb)
            skips.append(h)
        
        # Middle block (3D attention)
        h = self.mid_block(h, t, text_emb)
        
        # Upsample path
        for block in self.up_blocks:
            h = block(h, skips.pop(), t, text_emb)
        
        return self.conv_out(h)
