import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

from gif import create_gif

# --- CLASSES DCGAN (CONVOLUCIONAIS) ---

# class Discriminator(nn.Module):
#     def __init__(self, channels_img):
#         super(Discriminator, self).__init__()
#         self.disc = nn.Sequential(
#             # Input: N x 3 x 64 x 64
#             nn.Conv2d(channels_img, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         return self.disc(x).view(-1, 1)

# Fonte: Miyato, T., et al. "Spectral Normalization for Generative Adversarial Networks" (ICLR 2018).
class Discriminator(nn.Module):
    def __init__(self, channels_img):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x 3 x 64 x 64
            # Spectral Norm em TODAS as camadas Conv2d do Discriminador
            nn.utils.spectral_norm(nn.Conv2d(channels_img, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
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
            nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0), #4x4
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),   #8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, channels_img, kernel_size=4, stride=2, padding=1),   #64x64
            nn.Tanh(), 
            # Output: N x 3 x 64 x 64
        )

    def forward(self, x):
        return self.gen(x)
    
class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.data = []
        self.transform = transform
        
        # Define the static transforms (done ONCE at startup)
        # We resize and normalize here to save CPU time during training
        self.static_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        # Define dynamic transforms (done EVERY EPOCH)
        # Random flip must happen at runtime, or it's not random
        self.dynamic_transform = transforms.RandomHorizontalFlip(p=0.5)

        print("Loading dataset into RAM... (This may take a few seconds)")
        # We use ImageFolder just to find/load the files
        temp_dataset = torchvision.datasets.ImageFolder(root=root)
        
        for img, _ in temp_dataset:
            # Apply static transform immediately
            processed_tensor = self.static_transform(img)
            self.data.append(processed_tensor)
            
        print(f"Loaded {len(self.data)} images into RAM.")

    def __getitem__(self, index):
        tensor = self.data[index]
        # Apply the random flip on the Tensor
        tensor = self.dynamic_transform(tensor)
        return tensor, 0 # Return dummy label

    def __len__(self):
        return len(self.data)

def initialize_weights(model):
    # Inicialização de pesos é CRÍTICA para DCGAN funcionar bem
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def main():
    # --- CONFIGURAÇÕES ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando DCGAN em: {device}")
    
    LR_G = 0.0002
    LR_D = 0.00005
    BATCH_SIZE = 128
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3
    Z_DIM = 100
    NUM_EPOCHS = 700
    BETA1 = 0.5

    # --- Dataset ---
    IMAGE_SIZE = 64 

    _output_dir = "results"
    _epoch_dir = os.path.join(_output_dir, "dcgan_epochs")
    _file_names = "dcgan_epoch_"
    _model_name = "dcgan_model_final"
    _graph_name = "dcgan_loss_graph"
    _dataset_dir = "./data_cats/train"
    os.makedirs(_output_dir, exist_ok=True)
    os.makedirs(_epoch_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Garante tamanho uniforme
        transforms.RandomHorizontalFlip(p=0.5),      # TRUQUE NOVO: Aumenta dados espelhando
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # ImageFolder exige a estrutura root/classe/imagem.jpg
    # Se suas imagens estão soltas em data_cats, mova para data_cats/images/
    # dataset = torchvision.datasets.ImageFolder(root="./data_cats/train", transform=transform)
    dataset = CachedDataset(root="./data_cats/train")
    dataset = CachedDataset(root=_dataset_dir)

    # Remova o filtro de índices "cat_indices" pois agora o dataset é só de gatos
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0, 
        pin_memory=True
    )

    # --- Inicialização ---
    gen = Generator(Z_DIM, CHANNELS_IMG).to(device)
    disc = Discriminator(CHANNELS_IMG).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LR_G, betas=(BETA1, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR_D, betas=(BETA1, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

    G_losses = []
    D_losses = []

    print("Iniciando Loop de Treinamento...")

    for epoch in range(NUM_EPOCHS):
        epoch_d_loss = 0
        epoch_g_loss = 0
        
        noise_factor = 0.1 * (1 - (epoch / NUM_EPOCHS))
        if noise_factor < 0: noise_factor = 0

        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            current_batch_size = real.shape[0]
            
            # --- Train Discriminator ---
            # Gera ruído para injetar nas imagens (Instance Noise)
            noise_img = torch.randn_like(real) * noise_factor
            
            # Aplica ruído na imagem Real
            real_noisy = real + noise_img # ruino na imagem real
            disc_real = disc(real_noisy).reshape(-1) # Passa a imagem ruidosa

            # Label Smoothing (Real): Mantém 0.9
            labels_real_smooth = torch.ones_like(disc_real) * 0.9
            lossD_real = criterion(disc_real, labels_real_smooth)

            # Gera Fake
            noise_z = torch.randn(current_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise_z)

            # ruido na imagem fake para o discriminador
            fake_noisy = fake.detach() + noise_img
            disc_fake = disc(fake_noisy).reshape(-1)

            # 3. Two-sided Label Smoothing (Fake) - # Isso impede o discriminador de ter certeza absoluta
            labels_fake_smooth = torch.rand_like(disc_fake) * 0.2
            lossD_fake = criterion(disc_fake, labels_fake_smooth)            
            
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
        # with torch.no_grad():
        #     fake = gen(fixed_noise)
        #     file_path = os.path.join(_output_dir, f"{_file_names}{epoch+1:03d}.png")
        #     torchvision.utils.save_image(fake, file_path, normalize=True, nrow=8)
        if (epoch + 1) % 7 == 0 or epoch == 0:
            with torch.no_grad():
                fake = gen(fixed_noise)
                file_path = os.path.join(_epoch_dir, f"{_file_names}{epoch+1:03d}.png")
                torchvision.utils.save_image(fake, file_path, normalize=True, nrow=8)

    print("Salvando o modelo final...")
    torch.save(gen.state_dict(), os.path.join(_output_dir, f"{_model_name}.pth"))

    print("Treinamento Finalizado.")
    create_gif(input_folder=_epoch_dir, output_gif=f"{_file_names}.gif", file_names=_file_names, duration=20)

    plt.figure(figsize=(10, 5))
    plt.title("Curva de Aprendizado: Gerador vs Discriminador")
    plt.plot(G_losses, label="Gerador (G)")
    plt.plot(D_losses, label="Discriminador (D)")
    plt.xlabel("Épocas")
    plt.ylabel("Loss (BCE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(_output_dir, f"{_graph_name}.png"))
    print(f"Gráfico de loss salvo em {_output_dir}/{_graph_name}.png")

if __name__ == "__main__":
    main()