import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
from gif import create_gif  # Assumindo que você tem esse arquivo

# --- CONFIGURAÇÕES ---
BATCH_SIZE = 128
IMAGE_SIZE = 32 
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 25
LR = 0.0002
BETA1 = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ARQUITETURA ---
class Discriminator(nn.Module):
    def __init__(self, channels_img):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: 1 x 32 x 32
            nn.Conv2d(channels_img, 64, kernel_size=4, stride=2, padding=1), # Saída: 16x16
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Saída: 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # Saída: 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Removemos uma camada intermediária aqui para parar em 4x4
            
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0), # Input 4x4 -> Saída 1x1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x).view(-1, 1)

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: Z_DIM x 1 x 1
            nn.ConvTranspose2d(channels_noise, 256, kernel_size=4, stride=1, padding=0), # Saída: 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # Saída: 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # Saída: 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, channels_img, kernel_size=4, stride=2, padding=1), # Saída: 32x32
            nn.Tanh(), 
        )

    def forward(self, x):
        return self.net(x)

def initialize_weights(model):
    # Inicialização conforme paper DCGAN (Radford et al.)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

def main():
    print(f"Device: {DEVICE}")
    
    # Transformações
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), # Upscale forçado
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Dataset MNIST
    dataset = torchvision.datasets.MNIST(root=".", transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    gen = Generator(Z_DIM, CHANNELS_IMG).to(DEVICE)
    disc = Discriminator(CHANNELS_IMG).to(DEVICE)
    
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(BETA1, 0.999))
    
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(DEVICE)
    
    _output_folder = "results_mnist_dcgan"
    _file_names = "epoch_"
    os.makedirs(_output_folder, exist_ok=True)

    print("Iniciando treinamento DCGAN Otimizada...")

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # Labels para Soft Smoothing
    real_label_val = 0.9
    fake_label_val = 0.0


    for epoch in range(NUM_EPOCHS):
        for i, (real, _) in enumerate(dataloader):
            real = real.to(DEVICE)
            batch_len = real.shape[0]
            
            # --- TREINO DISCRIMINATOR ---
            disc.zero_grad()
            
            # Real (+ Soft Label Smoothing)
            # Labels reais são 0.9
            label_real = torch.full((batch_len,), real_label_val, dtype=torch.float, device=DEVICE)
            output_real = disc(real).view(-1)
            lossD_real = criterion(output_real, label_real)
            lossD_real.backward()
            
            # Fake
            noise = torch.randn(batch_len, Z_DIM, 1, 1).to(DEVICE)
            fake = gen(noise)
            label_fake = torch.full((batch_len,), fake_label_val, dtype=torch.float, device=DEVICE)
            
            # .detach() é crucial aqui para não treinar o Gerador nesta etapa
            output_fake = disc(fake.detach()).view(-1) 
            lossD_fake = criterion(output_fake, label_fake)
            lossD_fake.backward()
            
            lossD = lossD_real + lossD_fake
            opt_disc.step()

            # --- TREINO GENERATOR ---
            gen.zero_grad()
            # O gerador quer enganar o discriminador, então usamos label REAL (1.0 ou 0.9)
            # Mas para o loss do gerador, costuma-se usar 1.0 para gradientes mais fortes
            label_gen = torch.full((batch_len,), 1.0, dtype=torch.float, device=DEVICE)
            output = disc(fake).view(-1)
            lossG = criterion(output, label_gen)
            lossG.backward()
            opt_gen.step()

            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {i}/{len(dataloader)} Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

        # Salvar imagens
        with torch.no_grad():
            fake = gen(fixed_noise)
            vutils.save_image(
                fake,
                f"{_output_folder}/{_file_names}{epoch+1:03d}.png",
                normalize=True,
                nrow=8,
                padding=2
            )

    print("Treinamento finalizado.")
    create_gif(input_folder=_output_folder, output_gif=f"evolution.gif", file_names=_file_names, duration=500)

    plt.figure(figsize=(10,5))
    plt.title("Progresso do Loss do Gerador e Discriminador")
    plt.plot(G_losses, label="Gerador (G)")
    plt.plot(D_losses, label="Discriminador (D)")
    plt.xlabel("Iterações")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{_output_folder}/loss_plot.png") # Salva o gráfico
    plt.close()
    print(f"Gráfico de loss salvo em {_output_folder}/loss_plot.png")

    create_gif(input_folder=_output_folder, output_gif=f"evolution.gif", file_names=_file_names, duration=4)

if __name__ == "__main__":
    main()