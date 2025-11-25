import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

from gif import create_gif

# --- CLASSES NO ESCOPO GLOBAL (Necessário para Windows) ---

class Discriminator(nn.Module):
    def __init__(self, flat_img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(flat_img_dim, 1024), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, flat_img_dim, channels, image_size):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, flat_img_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x).view(-1, self.channels, self.image_size, self.image_size)

def main():
    # --- CONFIGURAÇÕES ---
    print(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    lr = 0.0002
    batch_size = 128
    z_dim = 100
    epochs = 250 
    image_size = 32
    channels = 3
    flat_img_dim = image_size * image_size * channels

    # --- 1. Preparar Dataset (CIFAR-10 - GATOS) ---
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Baixa dataset na pasta data_cifar para não misturar
    full_dataset = torchvision.datasets.CIFAR10(root="./data_cifar", train=True, download=True, transform=transform)

    # Filtragem: Classe 3 = Gato
    print("Filtrando apenas gatos...")
    cat_indices = [i for i, (img, label) in enumerate(full_dataset) if label == 3]
    dataset = torch.utils.data.Subset(full_dataset, cat_indices)

    # Dataloader protegido dentro do main
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, # Multiprocessing ativado
        pin_memory=True
    )

    # Inicializar
    gen = Generator(z_dim, flat_img_dim, channels, image_size).to(device)
    disc = Discriminator(flat_img_dim).to(device)
    
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr*0.5, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # --- Configuração de salvamento ---
    _output_dir = "results_cats_mlp"
    _file_names = "cat_mlp_epoch_"
    os.makedirs(_output_dir, exist_ok=True)
    print(f"Iniciando treinamento. Imagens salvas em '{_output_dir}'.")

    fixed_noise = torch.randn((32, z_dim)).to(device)

    # --- Loop de Treinamento ---
    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            curr_batch_size = real.shape[0]

            ### Treinar Discriminador
            noise = torch.randn(curr_batch_size, z_dim).to(device)
            fake = gen(noise)
            
            # Achata a imagem real para o discriminador Linear
            real_flat = real.view(-1, flat_img_dim)
            fake_flat = fake.view(-1, flat_img_dim)

            # Labels com smoothing
            real_labels = torch.ones(curr_batch_size, 1).to(device) * 0.9 
            fake_labels = torch.zeros(curr_batch_size, 1).to(device) + 0.1

            disc_real = disc(real_flat)
            lossD_real = criterion(disc_real, real_labels)
            
            disc_fake = disc(fake_flat.detach())
            lossD_fake = criterion(disc_fake, fake_labels)
            
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            ### Treinar Gerador
            output = disc(fake_flat)
            lossG = criterion(output, torch.ones(curr_batch_size, 1).to(device))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss D: {lossD:.4f}, loss G: {lossG:.4f}")

        with torch.no_grad():
            fake_display = gen(fixed_noise)
            file_name = os.path.join(_output_dir, f"{_file_names}{epoch+1:03d}.png")
            torchvision.utils.save_image(fake_display, file_name, normalize=True, nrow=8)

    print("Treinamento finalizado.")
    create_gif(input_folder=_output_dir, output_gif=f"{_file_names}.gif", file_names=_file_names, duration=25)

if __name__ == "__main__":
    main()