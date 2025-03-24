import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

class VAE(nn.Module):
    def __init__(self, hidden_size: int = 784, n_layers: int = 4, middle_size: int = 128, classes: int = 10, cond_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.middle_size = middle_size
        self.n_classes = classes
        self.cond_size = cond_size

        block = lambda in_size, out_size: nn.Sequential(
             nn.RMSNorm(in_size, eps=1e-06),
             nn.GELU(approximate='tanh'),
             nn.Linear(in_size, out_size)
        )

        self.encoder = nn.Sequential(
            *[block(hidden_size, hidden_size) for l in range(n_layers // 2)],
            block(hidden_size, middle_size*2)
        )
        self.cond_er = block(cond_size, middle_size)
        self.decoder = nn.Sequential(
            block(middle_size, hidden_size),
            *[block(hidden_size, hidden_size) for l in range(n_layers // 2)]
        )
        self.class_embedding = nn.Embedding(classes + 1, cond_size)

    def forward(self, x, cond=None, eps=None):
        self.class_embedding.weight[0].detach().zero_()
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = mu + (torch.randn_like(mu) if eps is None else eps.to(mu)) * (0.5 * logvar).exp()
        x = self.decode(z, cond)
        return x, mu, logvar
    
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.split(h, self.middle_size, -1)
        return mu, logvar
    
    def decode(self, z, cond=None):
        if cond is not None:
            z = z + self.cond_er(cond)
        x = self.decoder(z)
        return x.view(x.shape[0], 1, 28, 28)

    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()
        x_hat, mu, logvar = self.forward(x, self.class_embedding(y))
        loss_recon = nn.functional.mse_loss(x_hat, x)
        loss_kl = -0.5 * (1 - mu.square() + logvar - logvar.exp()).sum(1).mean()
        loss = loss_recon + loss_kl
        loss.backward()
        optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def sample(self, batch_size=1, y=None):
        if y is None:
            y = torch.randint(0, self.n_classes, (batch_size,), device=self.device)
        z = torch.randn(batch_size, self.middle_size).to(next(self.parameters()))
        x = self.decode(z, self.class_embedding(y))
        return x

    @property
    def device(self):
        return next(self.parameters()).device


def train_and_sample(epochs=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)

    vae = VAE().to(device)
    optimizer = torch.optim.AdamW(vae.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(dataloader), epochs=epochs)

    losses = []
    for epoch in (pbar:=tqdm(range(1, epochs + 1))):
        for batch_idx, (images, classes) in enumerate(tqdm(dataloader, leave=False)):
            losses.append(vae.train_step(images.to(device), ((classes + 1) * (torch.rand(classes.shape) < 0.8)).to(device), optimizer))
            scheduler.step()
            pbar.set_description(f'Epoch {epoch}, Batch {batch_idx}, Loss: {losses[-1]:.4f}, LR: {scheduler.get_last_lr()[-1]:.2e}')

        samples = vae.sample(121, torch.arange(vae.n_classes+1,).repeat(121 // vae.n_classes)[:121].to(device)).clamp(-1, 1)
        save_image(make_grid(samples, vae.n_classes+1, normalize=True, value_range=(-1, 1)), f'vae_samples_epoch_{epoch}.png')

    return vae, losses

if __name__ == "__main__":
    model, losses = train_and_sample()
