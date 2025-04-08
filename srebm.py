import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

class SREBM(nn.Module):
    def __init__(self, hidden_size: int = 784, n_layers: int = 8, middle_size: int = 2048, classes: int = 10, infer_steps: int = 4, optim_lr: float = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.middle_size = middle_size
        self.n_classes = classes
        self.infer_steps = infer_steps
        self.optim_lr = optim_lr

        self.ebm = nn.Sequential(
             nn.utils.spectral_norm(nn.Linear(hidden_size, middle_size)),
             nn.GELU(approximate='tanh'),
             *[nn.Sequential(nn.utils.spectral_norm(nn.Linear(middle_size, middle_size)), nn.GELU(approximate='tanh')) for l in range(n_layers - 1)],
             nn.utils.spectral_norm(nn.Linear(middle_size, classes + 1))
        )

    def forward(self, x):
        return self.ebm(x)
    
    def infer_x(self, y, z):
        z.detach_().requires_grad_(True)
        optim_inference = torch.optim.AdamW([z], lr=self.optim_lr)
        for inference_step in range(self.infer_steps):
            optim_inference.zero_grad()
            energy = self.forward(z)[range(z.shape[0]), y].sum()
            energy.backward()
            optim_inference.step()
        return z.detach()

    def train_step(self, x, y, optimizer):
        x = x.view(-1, 784)
        x = x + torch.randn_like(x) * 0.01
        z_0 = torch.randn_like(x).uniform_(-1, 1)
        z = self.infer_x(y, z_0)
        pos_energy, neg_energy = self.forward(x)[range(x.shape[0]), y].mean(), self.forward(z)[range(x.shape[0]), y].mean()
        loss = pos_energy - neg_energy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def sample(self, batch_size=1, y=None):
        if y is None:
            y = torch.randint(0, self.n_classes, (batch_size,), device=self.device)
        z = torch.randn(batch_size, 784, requires_grad=False, device=self.device).uniform_(-1, 1).requires_grad_(True)
        z = self.infer_x(y, z)
        return z.view(batch_size, 1, 28, 28)

    @property
    def device(self):
        return next(self.parameters()).device


def train_and_sample(epochs=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)

    srebm = SREBM().to(device)
    print(f'{sum(p.numel() for p in srebm.parameters()):,} parameters.')
    optimizer = torch.optim.AdamW(srebm.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(dataloader), epochs=epochs)

    losses = []
    for epoch in (pbar:=tqdm(range(1, epochs + 1))):
        for batch_idx, (images, classes) in enumerate(tqdm(dataloader, leave=False)):
            losses.append(srebm.train_step(images.to(device), ((classes + 1) * (torch.rand(classes.shape) < 0.8)).to(device), optimizer))
            scheduler.step()
            pbar.set_description(f'Epoch {epoch}, Batch {batch_idx}, Loss: {losses[-1]:.2e}, LR: {scheduler.get_last_lr()[-1]:.2e}')

        samples = srebm.sample(121, torch.arange(srebm.n_classes+1,).repeat(121 // srebm.n_classes)[:121].to(device)).clamp(-1, 1)
        save_image(make_grid(samples, srebm.n_classes+1, normalize=True, value_range=(-1, 1)), f'srebm_samples_epoch_{epoch}.png')

    return srebm, losses

if __name__ == "__main__":
    model, losses = train_and_sample()
