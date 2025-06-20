import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

class GAN(nn.Module):
    def __init__(self, hidden_size: int = 784, middle_size: int = 128, classes: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.middle_size = middle_size
        self.n_classes = classes

        self.discriminator = nn.Sequential(
            nn.Linear(hidden_size, 4 * middle_size),
            nn.GELU(approximate='tanh'),
            nn.Dropout(),
            nn.Linear(4 * middle_size, 2 * middle_size),
            nn.GELU(approximate='tanh'),
            nn.Dropout(),
            nn.Linear(2 * middle_size, 1),
            nn.Sigmoid()
        )
        self.generator = nn.Sequential(
            nn.Linear(middle_size, 2 * middle_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(2 * middle_size, 3 * middle_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(3 * middle_size, 4 * middle_size),
            nn.RMSNorm(4 * middle_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * middle_size, hidden_size),
            nn.Tanh()
        )

        self.class_embedding_g = nn.Embedding(classes + 1, middle_size)
        self.class_embedding_d = nn.Embedding(classes + 1, hidden_size)

    def forward(self, z, cond=None):
        with torch.no_grad():
            self.class_embedding_g.weight[0].zero_()
            self.class_embedding_d.weight[0].zero_()
        x = self.decode(z, cond)
        return x

    def decode(self, z, cond=None):
        if cond is not None:
            z = z + cond
        x = self.generator(z)
        return x.view(x.shape[0], 1, 28, 28)

    def classify(self, x, cond=None):
        x = x.view(x.shape[0], -1)
        if cond is not None:
            x = x + cond
        return self.discriminator(x)

    def train_step(self, x, y, z, optimizer):
        optimizer.zero_grad()
        y_fake = torch.randint_like(y, 0, self.n_classes+1)
        x_hat = self.forward(z, self.embed_class(y_fake))
        loss_discriminator_real = nn.functional.mse_loss(self.classify(x, self.embed_class(y, g=False)), torch.ones(x.shape[0], 1, device=self.device))
        loss_discriminator_fake = nn.functional.mse_loss(self.classify(x_hat.detach(), self.embed_class(y_fake, g=False)), torch.zeros(x.shape[0], 1, device=self.device))
        (0.5 * (loss_discriminator_real + loss_discriminator_fake)).backward()
        loss_generator = nn.functional.mse_loss(self.classify(x_hat, self.embed_class(y_fake, g=False)), torch.ones(x.shape[0], 1, device=self.device))
        loss_generator.backward()
        optimizer.step()
        return (loss_generator.item(), (loss_discriminator_real + loss_discriminator_fake).item())
    
    def embed_class(self, y, g=True):
        if y is None:
            return y
        elif g:
            return self.class_embedding_g(y.squeeze().long())
        else:
            return self.class_embedding_d(y.squeeze().long())

    @torch.no_grad()
    def sample(self, batch_size=1, y=None):
        if y is None:
            y = torch.randint(0, self.n_classes, (batch_size,), device=self.device)
        z = torch.randn(batch_size, self.middle_size).to(next(self.parameters()))
        x = self.decode(z, self.embed_class(y))
        return x.view(x.shape[0], 1, 28, 28)

    @property
    def device(self):
        return next(self.parameters()).device


def train_and_sample(epochs=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    gan = GAN().to(device)
    optimizer = torch.optim.AdamW(gan.parameters(), betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=len(dataloader), epochs=epochs)

    losses = []
    for epoch in (pbar:=tqdm(range(1, epochs + 1))):
        for batch_idx, (images, classes) in enumerate(tqdm(dataloader, leave=False)):
            gan.train()
            losses.append(gan.train_step(images.to(device), ((classes + 1) * (torch.rand(classes.shape) < 0.8)).to(device), torch.randn(images.shape[0], gan.middle_size).to(device), optimizer))
            scheduler.step()
            pbar.set_description(f'Epoch {epoch}, Batch {batch_idx}, Loss: {losses[-1][0]:.4f}G+{losses[-1][1]:.4f}D, LR: {scheduler.get_last_lr()[-1]:.2e}')
        
        gan.eval()
        samples = gan.sample(121, torch.arange(gan.n_classes+1,).repeat(121 // gan.n_classes)[:121].to(device)).clamp(-1, 1)
        save_image(make_grid(samples, gan.n_classes+1, normalize=True, value_range=(-1, 1)), f'gan_samples_epoch_{epoch}.png')

    return gan, losses

if __name__ == "__main__":
    model, losses = train_and_sample()