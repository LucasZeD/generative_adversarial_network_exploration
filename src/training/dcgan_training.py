import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from pathlib import Path
import torch.nn as nn
from PIL import Image
import torchvision
import torch
import sys
import os

plt.switch_backend('Agg')

FILE_PATH = Path(__file__).resolve()
# src/training/dcgan_training.py -> parents[0]=training, parents[1]=src, parents[2]=root
PROJECT_ROOT = FILE_PATH.parents[2]
sys.path.append(str(PROJECT_ROOT))

try:
    from src.utils.gif import create_gif
except ImportError:
    print("AVISO: src.utils.gif não encontrado. A função create_gif será ignorada.")
    create_gif = None

# --- CLASSES DCGAN ---
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
        # Dynamic transform apenas para Augmentation (Flip)
        self.dynamic_transform = transforms.RandomHorizontalFlip(p=0.5)
        
        # Static Transform: Resize + Normalize roda UMA VEZ na carga
        self.static_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        print(f"Carregando dataset de {root} para RAM...")
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset não encontrado em: {root}")
        
        # Tenta carregar usando ImageFolder (espera subpastas)
        try:
            temp_dataset = torchvision.datasets.ImageFolder(root=str(root))
            print("Estrutura de pastas de classe detectada.")
            for img, _ in temp_dataset:
                processed_tensor = self.static_transform(img)
                self.data.append(processed_tensor)
        
        # Se falhar (erro de class folder), carrega arquivos soltos (Fallback)
        except Exception:
            print("Aviso: Nenhuma subpasta encontrada. Alternando para modo 'Flat Directory'...")
            
            # Pega extensões comuns de imagem
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
            files = []
            for ext in extensions:
                files.extend(list(root_path.glob(ext)))
                # Tenta versão maiúscula também
                files.extend(list(root_path.glob(ext.upper())))
            
            if not files:
                raise FileNotFoundError(f"Nenhuma imagem encontrada em {root}")

            for file_path in files:
                try:
                    # Carrega imagem manualmente com PIL
                    img = Image.open(file_path).convert("RGB")
                    processed_tensor = self.static_transform(img)
                    self.data.append(processed_tensor)
                except Exception as e:
                    print(f"Erro ao ler {file_path.name}: {e}")

        print(f"Carregado: {len(self.data)} imagens.")

    def __getitem__(self, index):
        tensor = self.data[index]
        # Apply the random flip on the Tensor
        tensor = self.dynamic_transform(tensor)
        return tensor, 0 # Return dummy label

    def __len__(self):
        return len(self.data)

def initialize_weights(model):
    # Critico! Inicialização de pesos
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando DCGAN em: {device}")
    
    # Hyperparâmetros
    LR_G = 0.0002
    LR_D = 0.00005
    BATCH_SIZE = 128
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3
    Z_DIM = 100
    NUM_EPOCHS = 700
    BETA1 = 0.5

    # Early stopping
    MIN_D_LOSS = 0.05 # Se o D ficar muito bom, o G não aprende mais
    PATIENCE = 50 # Épocas consecutivas permitidas abaixo do limiar
    patience_counter = 0

    # Absolute paths e criação de pastas
    DATASET_DIR = PROJECT_ROOT / "data" / "train"
    RESULTS_DIR = PROJECT_ROOT / "results"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    LOGS_DIR = RESULTS_DIR / "training_logs" / "dcgan_epochs"

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    GRAPH_PATH = RESULTS_DIR / "dcgan_loss_graph.png"
    MODEL_PATH = CHECKPOINTS_DIR / "dcgan_model_final.pth"

    # --- Dataset ---
    try:
        # ImageFolder exige a estrutura root/classe/imagem.jpg
        dataset = CachedDataset(root=DATASET_DIR)
    except Exception as e:
        print(f"ERRO FATAL: {e}")
        return

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0, # windows/cached dataset não funciona bem com múltiplos workers
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

    try:
        for epoch in range(NUM_EPOCHS):
            epoch_d_loss = 0
            epoch_g_loss = 0
            
            # Noise Decay: Reduz o ruído injetado conforme o treino avança
            noise_factor = 0.1 * (1 - (epoch / NUM_EPOCHS))
            noise_factor = max(0, noise_factor)

            for batch_idx, (real, _) in enumerate(loader):
                real = real.to(device)
                current_batch_size = real.shape[0]
                
                # --- 1. Treinar Discriminador ---
                noise_img = torch.randn_like(real) * noise_factor
                real_noisy = real + noise_img
                
                disc_real = disc(real_noisy).reshape(-1)
                labels_real = torch.ones_like(disc_real) * 0.9 # Label Smoothing
                lossD_real = criterion(disc_real, labels_real)

                noise_z = torch.randn(current_batch_size, Z_DIM, 1, 1).to(device)
                fake = gen(noise_z)
                
                fake_noisy = fake.detach() + noise_img
                disc_fake = disc(fake_noisy).reshape(-1)
                labels_fake = torch.rand_like(disc_fake) * 0.2 # Two-sided Smoothing
                lossD_fake = criterion(disc_fake, labels_fake)            
                
                lossD = (lossD_real + lossD_fake) / 2
                disc.zero_grad()
                lossD.backward()
                opt_disc.step()

                # --- 2. Treinar Gerador ---
                output = disc(fake).reshape(-1)
                lossG = criterion(output, torch.ones_like(output))

                gen.zero_grad()
                lossG.backward()
                opt_gen.step()

                epoch_d_loss += lossD.item()
                epoch_g_loss += lossG.item()

            # Médias
            avg_d_loss = epoch_d_loss / len(loader)
            avg_g_loss = epoch_g_loss / len(loader)
            G_losses.append(avg_g_loss)
            D_losses.append(avg_d_loss)

            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss D: {avg_d_loss:.4f}, Loss G: {avg_g_loss:.4f} | Noise: {noise_factor:.3f}")

            # --- EARLY STOPPING CHECK ---
            if avg_d_loss < MIN_D_LOSS:
                patience_counter += 1
                print(f" ALERTA: Discriminador muito forte ({avg_d_loss:.4f}). Patience {patience_counter}/{PATIENCE}")
                if patience_counter >= PATIENCE:
                    print(" PARADA ANTECIPADA: Discriminador venceu (Overfitting/Mode Collapse).")
                    break
            else:
                patience_counter = 0 # Reseta se recuperar

            # Logs visuais
            if (epoch + 1) % 5 == 0 or epoch == 0:
                with torch.no_grad():
                    fake_display = gen(fixed_noise)
                    # Caminho agora usa operador / corretamente com objetos Path
                    file_path = LOGS_DIR / f"epoch_{epoch+1:03d}.png"
                    torchvision.utils.save_image(fake_display, str(file_path), normalize=True, nrow=8)
            
            # Salvar checkpoint periodicamente (a cada 50 épocas)
            if (epoch + 1) % 50 == 0:
                 torch.save(gen.state_dict(), CHECKPOINTS_DIR / f"dcgan_epoch_{epoch+1}.pth")
    except KeyboardInterrupt:
        print("\nInterrupção manual detectada. Salvando estado atual...")

    print(f"Salvando modelo final em {MODEL_PATH}...")
    torch.save(gen.state_dict(), str(MODEL_PATH))

    if create_gif:
        GIF_PATH = RESULTS_DIR / "dcgan_training_evolution.gif"
        # Converte Paths para string para a função externa
        create_gif(input_folder=str(LOGS_DIR), output_gif=str(GIF_PATH), file_names="epoch_", duration=20)

    plt.figure(figsize=(10, 5))
    plt.title("Learning Curve: Gerador vs Discriminador")
    plt.plot(G_losses, label="Gerador (G)")
    plt.plot(D_losses, label="Discriminador (D)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (BCE)")
    plt.legend()
    plt.savefig(str(GRAPH_PATH))
    plt.close()
    print(f"Gráfico salvo em {GRAPH_PATH}")

if __name__ == "__main__":
    main()