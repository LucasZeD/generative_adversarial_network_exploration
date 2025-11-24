import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from gif import create_gif  # assuming you have this function

# === Models (must be at global scope for Windows multiprocessing) ===
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(784, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),   nn.Sigmoid(),
        )
    def forward(self, x):
        return self.disc(x.view(-1, 784))

class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),   nn.ReLU(),
            nn.Linear(256, 784),   nn.Tanh(),
        )
    def forward(self, x):
        return self.gen(x).view(-1, 1, 28, 28)

def main():
    # --- FORCE CPU ONLY ---
    device = torch.device("cpu")
    print(f"Usando dispositivo: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # --- CONFIGURAÇÕES OTIMIZADAS PARA CPU ---
    lr         = 0.0002
    batch_size = 128
    z_dim      = 100
    epochs     = 60

    # Dataset MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.MNIST(
        root="./mnist_data",   # separate folder so it doesn't clutter
        train=True,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # Models & optimizers
    gen  = Generator(z_dim).to(device)
    disc = Discriminator().to(device)

    opt_gen  = optim.Adam(gen.parameters(),  lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Fixed noise for consistent sampling during training
    fixed_noise = torch.randn(32, z_dim).to(device)

    # Results folder
    _input_folder = "results_cpu"
    _file_names = "gan_generated_epoch_"

    os.makedirs(_input_folder, exist_ok=True)
    print("Iniciando treinamento (CPU only)...")

    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            cur_bs = real.shape[0]

            # -----------------
            #  Train Discriminator
            # -----------------
            noise = torch.randn(cur_bs, z_dim).to(device)
            fake = gen(noise)

            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))

            disc_fake = disc(fake.detach()).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            # -----------------
            #  Train Generator
            # -----------------
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

        # --- Epoch end: log + save image ---
        print(f"Epoch [{epoch+1:02d}/{epochs}]  Loss D: {lossD:.4f}  Loss G: {lossG:.4f}")

        with torch.no_grad():
            fake_imgs = gen(fixed_noise)
            save_path = f"{_input_folder}/{_file_names}{epoch+1:03d}.png"
            torchvision.utils.save_image(
                fake_imgs, save_path,
                nrow=8, normalize=True, padding=2
            )

    print("Treinamento concluído!")
    create_gif(input_folder=_input_folder, output_gif=f"{_file_names}.gif", file_names=_file_names, duration=6)
# ==============================================================================
if __name__ == "__main__":
    main()