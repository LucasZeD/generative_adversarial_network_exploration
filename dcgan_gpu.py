import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import torchvision.utils as vutils

from gif import create_gif

# --- CONFIGURAÇÕES ---
# Batch size 128 é o "sweet spot" para qualidade e estabilidade
BATCH_SIZE = 128 
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 20
LR = 0.0002
BETA1 = 0.5

# --- ARQUITETURA DCGAN (CONVOLUCIONAL) ---
# É isso que vai fazer sua GPU trabalhar de verdade.

class Discriminator(nn.Module):
    def __init__(self, channels_img):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x).view(-1, 1)

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            nn.ConvTranspose2d(channels_noise, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), 
            # Saída: N x channels_img x 64 x 64
        )

    def forward(self, x):
        return self.net(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def main():
    print(f"PyTorch Version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Resize para 64x64 é essencial para esta arquitetura
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = torchvision.datasets.MNIST(root=".", transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    gen = Generator(Z_DIM, CHANNELS_IMG).to(device)
    disc = Discriminator(CHANNELS_IMG).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(BETA1, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
    
    _input_folder = "results_dcgan_gpu"
    _file_names = "dcgan_epoch_"

    os.makedirs(_input_folder, exist_ok=True)
    print("Iniciando treinamento DCGAN (GPU Heavy)...")

    for epoch in range(NUM_EPOCHS):
        for i, (real, _) in enumerate(dataloader):
            real = real.to(device)
            noise = torch.randn(real.shape[0], Z_DIM, 1, 1).to(device)
            
            # --- Treino Discriminator ---
            fake = gen(noise)
            disc_real = disc(real).reshape(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            # --- Treino Generator ---
            output = disc(fake).reshape(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()
            
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss D: {lossD:.4f}, Loss G: {lossG:.4f}")

        with torch.no_grad():
            fake = gen(fixed_noise)
            # Normalizar para visualização correta
            vutils.save_image(
                fake,
                f"{_input_folder}/{_file_names}{epoch+1:03d}.png",
                normalize=True,
                nrow=8,          # deixa 8 imagens por linha → grid bonito
                padding=2
            )

    print("Finalizado.")
    create_gif(input_folder=_input_folder, output_gif=f"{_file_names}.gif", file_names=_file_names, duration=2)
if __name__ == "__main__":
    main()