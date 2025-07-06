class AdaptiveNoiseScheduler:
    def __init__(self, max_steps=1000):
        self.max_steps = max_steps
        self.betas = torch.linspace(1e-6, 0.02, max_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x, noise, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t])
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
    
    def step(self, pred_noise, t, x):
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        
        pred_x0 = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        x_prev = sqrt_alpha_t * pred_x0 + sqrt_one_minus_alpha_t * pred_noise
        
        return x_prev
