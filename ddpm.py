import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

class ResNet(nn.Module):
    def __init__(self, hidden_size: int = 784, n_layers: int = 4, middle_size: int = 2048, cond_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.middle_size = middle_size
        self.cond_size = cond_size

        block = lambda in_size, out_size: nn.Sequential(
             nn.RMSNorm(in_size, eps=1e-06),
             nn.GELU(approximate='tanh'),
             nn.Linear(in_size, out_size)
        )

        self.blocks = nn.ModuleList([nn.ModuleDict({
            'block_head': block(hidden_size, middle_size),
            'block_cond': nn.Linear(cond_size, middle_size, bias=False),
            'block_tail': block(middle_size, hidden_size)
        }) for l in range(n_layers)])

        with torch.no_grad():
            for block in self.blocks:
                block.block_tail[-1].weight.zero_()
                block.block_tail[-1].bias.zero_()

    def forward(self, x, cond=None):
        assert x.shape[-1] == self.hidden_size, "Input must be hidden size."
        for block in self.blocks:
            h = block['block_head'](x)
            if cond is not None:
                h = h + block['block_cond'](cond)
            x = x + block['block_tail'](h)
        return x


class MinimalDDPM(nn.Module):
    def __init__(self, timesteps=1000, classes=10, time_embed_size=128, class_embed_size=128):
        super().__init__()
        self.model = ResNet()
        self.time_embedding = nn.Embedding(timesteps, time_embed_size)
        self.class_embedding = nn.Embedding(classes + 1, class_embed_size)
        self.timesteps = timesteps
        self.n_classes = classes
        betas = torch.linspace(0.0001, 0.02, timesteps)
        self.register_buffer('alphas', 1.0 - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def forward(self, x, t, y):
        self.class_embedding.weight[0].detach().zero_()
        x = x.view(-1, 784)
        t = self.time_embedding(t)
        y = self.class_embedding(y)
        ty = torch.concat((t, y), dim=1)
        return self.model(x, ty).view(-1, 1, 28, 28)

    def add_noise(self, x_0, t):
        noise = torch.randn_like(x_0)
        alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1 - alpha_cumprod_t) * noise, noise

    def train_step(self, x_0, y, optimizer):
        optimizer.zero_grad()
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=self.device)
        x_t, noise = self.add_noise(x_0, t)
        noise_pred = self.forward(x_t, t, y)
        loss = nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def sample(self, batch_size=1, y=None, guidance_weight=5.):
        x_t = torch.randn(batch_size, 1, 28, 28, device=self.device)
        if y is None:
            y = torch.randint(0, self.n_classes, (batch_size,), device=self.device)
        for t in tqdm(range(self.timesteps-1, -1, -1), desc='sampling', leave=False):
            t_tensor = torch.ones(batch_size, dtype=torch.long, device=self.device) * t
            noise_pred = self.forward(x_t, t_tensor, y)
            if guidance_weight:
                noise_uncond_pred = self.forward(x_t, t_tensor, torch.zeros_like(y))
                noise_pred = noise_uncond_pred + guidance_weight * (noise_pred - noise_uncond_pred)
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            noise = torch.randn_like(x_t) if t > 0 else 0
            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (1-alpha_t)/torch.sqrt(1-alpha_cumprod_t) * noise_pred) + torch.sqrt(1-alpha_t) * noise

        return x_t

    @property
    def device(self):
        return next(self.parameters()).device


def train_and_sample(epochs=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)

    ddpm = MinimalDDPM().to(device)
    optimizer = torch.optim.AdamW(ddpm.model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=len(dataloader), epochs=epochs)

    losses = []
    for epoch in (pbar:=tqdm(range(1, epochs + 1))):
        for batch_idx, (images, classes) in enumerate(tqdm(dataloader, leave=False)):
            losses.append(ddpm.train_step(images.to(device), ((classes + 1) * (torch.rand(classes.shape) < 0.8)).to(device), optimizer))
            scheduler.step()
            pbar.set_description(f'Epoch {epoch}, Batch {batch_idx}, Loss: {losses[-1]:.4f}, LR: {scheduler.get_last_lr()[-1]:.2e}')

        samples = ddpm.sample(121, torch.arange(ddpm.n_classes+1,).repeat(121 // ddpm.n_classes)[:121].to(device)).clamp(-1, 1)
        save_image(make_grid(samples, ddpm.n_classes+1, normalize=True, value_range=(-1, 1)), f'ddpm_samples_epoch_{epoch}.png')

    return ddpm, losses

if __name__ == "__main__":
    model, losses = train_and_sample()
