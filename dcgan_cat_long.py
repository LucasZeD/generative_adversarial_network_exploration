import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

from gif import create_gif

# --- CLASSES DCGAN (CONVOLUCIONAIS) ---

class Discriminator(nn.Module):
    def __init__(self, channels_img):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x 3 x 64 x 64
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
    def __init__(self, z_dim, channels_img):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0),
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
            # Output: N x 3 x 64 x 64
        )

    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    # Inicialização de pesos é CRÍTICA para DCGAN funcionar bem
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def main():
    # --- CONFIGURAÇÕES ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando DCGAN em: {device}")
    
    LEARNING_RATE = 0.0002
    BATCH_SIZE = 128
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3
    Z_DIM = 100
    NUM_EPOCHS = 2600
    BETA1 = 0.5

    # --- Dataset ---
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    full_dataset = torchvision.datasets.CIFAR10(root="./data_cifar", train=True, download=True, transform=transform)
    
    print("Filtrando dataset para manter apenas gatos...")
    cat_indices = [i for i, (img, label) in enumerate(full_dataset) if label == 3]
    dataset = torch.utils.data.Subset(full_dataset, cat_indices)

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )

    # --- Inicialização ---
    gen = Generator(Z_DIM, CHANNELS_IMG).to(device)
    disc = Discriminator(CHANNELS_IMG).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
    
    _output_dir = "results_cats_dcgan_long"
    _file_names = "dcgan_cat_epoch_"
    os.makedirs(_output_dir, exist_ok=True)

    G_losses = []
    D_losses = []

    print("Iniciando Loop de Treinamento...")

    for epoch in range(NUM_EPOCHS):
        epoch_d_loss = 0
        epoch_g_loss = 0
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            current_batch_size = real.shape[0]
            
            # --- Train Discriminator ---
            noise = torch.randn(current_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            
            disc_real = disc(real).reshape(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            
            disc_fake = disc(fake.detach()).reshape(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            # --- Train Generator ---
            output = disc(fake).reshape(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            epoch_d_loss += lossD.item()
            epoch_g_loss += lossG.item()
        avg_d_loss = epoch_d_loss / len(loader)
        avg_g_loss = epoch_g_loss / len(loader)
        G_losses.append(avg_g_loss)
        D_losses.append(avg_d_loss)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss D: {lossD:.4f}, Loss G: {lossG:.4f}")
        with torch.no_grad():
            fake = gen(fixed_noise)
            file_path = os.path.join(_output_dir, f"{_file_names}{epoch+1:03d}.png")
            torchvision.utils.save_image(fake, file_path, normalize=True, nrow=8)

    print("Treinamento Finalizado.")
    create_gif(input_folder=_output_dir, output_gif=f"{_file_names}.gif", file_names=_file_names, duration=20)

    plt.figure(figsize=(10, 5))
    plt.title("Curva de Aprendizado: Gerador vs Discriminador")
    plt.plot(G_losses, label="Gerador (G)")
    plt.plot(D_losses, label="Discriminador (D)")
    plt.xlabel("Épocas")
    plt.ylabel("Loss (BCE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(_output_dir, "loss_graph.png"))
    print(f"Gráfico de loss salvo em {_output_dir}/loss_graph.png")

if __name__ == "__main__":
    main()