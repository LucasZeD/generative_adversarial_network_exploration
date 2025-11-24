import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from gif import create_gif

# Definições de Classes devem ficar no escopo global para o multiprocessing funcionar
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(784, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.disc(x.view(-1, 784))

class Generator(nn.Module,):
    def __init__(self, z_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Tanh(),
        )
    def forward(self, x):
        return self.gen(x).view(-1, 1, 28, 28)

def main():
    # --- CONFIGURAÇÕES ---
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lr = 0.0002
    batch_size = 128
    z_dim = 100
    epochs = 60

    # Preparar Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(root=".", transform=transform, download=True)
    
    # DataLoader dentro da função main/guard
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4, # Agora isso funcionará no Windows
        pin_memory=True
    )

    # Inicializar
    gen = Generator(z_dim).to(device)
    disc = Discriminator().to(device)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # --- LOOP DE TREINAMENTO ---
    _input_folder = "results_gpu"
    _file_names = "gan_generated_epoch_"
    os.makedirs(_input_folder, exist_ok=True)
    print(f"Iniciando treinamento em {device}. As imagens serão salvas na pasta 'results'.")
    fixed_noise = torch.randn((32, z_dim)).to(device)

    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            curr_batch_size = real.shape[0]

            ### 1. Treinar Discriminador
            noise = torch.randn(curr_batch_size, z_dim).to(device)
            fake = gen(noise)
            
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            
            disc_fake = disc(fake.detach()).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            ### 2. Treinar Gerador
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

        # Logs e Salvamento
        print(f"Epoch [{epoch+1}/{epochs}] Loss D: {lossD:.4f}, loss G: {lossG:.4f}")

        with torch.no_grad():
            fake_display = gen(fixed_noise).reshape(-1, 1, 28, 28)
            file_name = f"{_input_folder}/{_file_names}{epoch+1:03d}.png"
            torchvision.utils.save_image(fake_display, file_name, normalize=True)
            # print(f"-> Imagem salva: {file_name}")

    print("Treinamento finalizado.")
    create_gif(input_folder=_input_folder, output_gif=f"{_file_names}.gif", file_names=_file_names, duration=6)
if __name__ == "__main__":
    # Esta linha impede que os processos filhos executem o código recursivamente
    main()