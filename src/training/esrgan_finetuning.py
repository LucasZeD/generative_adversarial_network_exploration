from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.optim as optim
from pathlib import Path
from PIL import Image
import torch.nn as nn
import torch
import sys
import os

plt.switch_backend('Agg')

FILE_PATH = Path(__file__).resolve()
# src/training/esrgan_finetuning.py -> parents[0]=training, parents[1]=src, parents[2]=root
PROJECT_ROOT = FILE_PATH.parents[2] 
sys.path.append(str(PROJECT_ROOT))

# --- 2. ARQUITETURA DO GERADOR (SRResNet) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, in_c * scale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()
    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))

class GeneratorSR(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=8):
        super(GeneratorSR, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.mid = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels)
        )
        self.upsamples = nn.Sequential(
            UpsampleBlock(num_channels, 2),
            UpsampleBlock(num_channels, 2),
            nn.Conv2d(num_channels, in_channels, kernel_size=9, padding=4)
        )
    def forward(self, x):
        initial = self.initial(x)
        res = self.residuals(initial)
        mid = self.mid(res) + initial
        return self.upsamples(mid)

# --- 3. PERDAS ---
class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Usa VGG19 até a camada 35 (relu5_4) para capturar texturas de alto nível
        print("Carregando VGG19 para Perceptual Loss...")
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:35].eval().to(device)
        
        # Congela pesos para não treinar a VGG
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        # Normalização manual compatível com AMP
        input_norm = (input - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        return self.mse(self.vgg(input_norm), self.vgg(target_norm))

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# --- 4. DATASET COM CACHE EM RAM ---
class CachedSRDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.data_high = []
        self.data_low = []
        
        # Transformações Base
        # High Res: 256x256 (Alvo)
        self.high_res_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # Normalização Opcional: Se sua DCGAN sai [-1,1], aqui deve ser igual
            # Mas VGG Loss espera [0,1] antes da normalização interna. 
            # SRGAN padrão geralmente opera em [0, 1] ou [-1, 1]. Vamos usar [0, 1] (ToTensor puro)
        ])
        
        # Low Res: 64x64 (Entrada)
        self.low_res_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        print(f"Carregando e processando dataset de {root_dir} para RAM...")
        root_path = Path(root_dir)
        if not root_path.exists():
             raise FileNotFoundError(f"Dataset não encontrado em: {root_dir}")
        
        # 1. Tenta carregar imagens manualmente (Flat Directory)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        files = []
        for ext in extensions:
            files.extend(list(root_path.glob(ext)))
            files.extend(list(root_path.glob(ext.upper()))) # Case insensitive check

        if len(files) > 0:
            print(f"Modo Flat Directory detectado. Encontrados {len(files)} arquivos.")
            for file_path in files:
                try:
                    img = Image.open(file_path).convert("RGB")
                    hr = self.high_res_transform(img)
                    lr = self.low_res_transform(img)
                    self.data_high.append(hr)
                    self.data_low.append(lr)
                except Exception as e:
                    print(f"Erro ao ler {file_path.name}: {e}")
                    
        else:
            # 2. Se não achou arquivos soltos, tenta ImageFolder (Subpastas)
            print("Nenhum arquivo solto encontrado. Tentando estrutura de pastas (ImageFolder)...")
            try:
                temp_dataset = datasets.ImageFolder(root=str(root_dir))
                for img, _ in temp_dataset:
                    hr = self.high_res_transform(img)
                    lr = self.low_res_transform(img)
                    self.data_high.append(hr)
                    self.data_low.append(lr)
            except Exception as e:
                raise FileNotFoundError(f"Não foi possível carregar imagens nem como Flat nem como Folder: {e}")
            
        print(f"Cache concluído: {len(self.data_high)} pares de imagens carregados.")

    def __len__(self):
        return len(self.data_high)

    def __getitem__(self, index):
        return self.data_low[index], self.data_high[index]

def main():
    if torch.cuda.is_available():
        # Acelera convs se o tamanho da img for fixo
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("AVISO: Rodando em CPU. Isso será extremamente lento.")
    
    print(f"=== Treinando SRGAN em: {device} ===")
    # Hiperparâmetros
    LR = 1e-4
    BATCH_SIZE = 16 # Se der OutOfMemory, baixe para 8 ou 4
    EPOCHS = 100
    
    # Caminhos
    DATASET_DIR = PROJECT_ROOT / "data" / "train"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    RESULTS_DIR = PROJECT_ROOT / "results"
    OUTPUT_IMGS_DIR = RESULTS_DIR / "training_logs" / "srgan_epochs"

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMGS_DIR.mkdir(parents=True, exist_ok=True)
    
    MODEL_PATH = CHECKPOINTS_DIR / "sr_generator_finetuned.pth"
    LOSS_GRAPH_PATH = RESULTS_DIR / "srgan_loss_graph.png"

    # Dataset e Loader
    try:
        dataset = CachedSRDataset(root_dir=DATASET_DIR)
    except Exception as e:
        print(f"ERRO: {e}")
        return

    # Com dataset em RAM, num_workers=0 é mais rápido (sem overhead de multiprocessamento)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # Inicialização
    gen = GeneratorSR().to(device)
    opt_gen = optim.Adam(gen.parameters(), lr=LR)
    
    mse_loss = nn.MSELoss()
    vgg_loss = VGGLoss(device).to(device)
    tv_loss = TVLoss(tv_loss_weight=1e-5).to(device)

    scaler = GradScaler()
    loss_history = []

    print("Iniciando Treinamento...")

    try:
        for epoch in range(EPOCHS):
            epoch_loss = 0
            
            for idx, (low_res, high_res) in enumerate(loader):
                low_res = low_res.to(device, non_blocking=True)
                high_res = high_res.to(device, non_blocking=True)

                opt_gen.zero_grad()

                # Autocast para float16 onde possível
                with autocast():
                    fake_high_res = gen(low_res)
                    
                    # Cálculo da Loss Híbrida
                    l_mse = mse_loss(fake_high_res, high_res)
                    l_vgg = vgg_loss(fake_high_res, high_res)
                    l_tv = tv_loss(fake_high_res)
                    
                    # Pesos ajustados:
                    # 1.0 MSE (Estrutura básica)
                    # 0.006 VGG (Textura fina - o 0.006 compensa a magnitude da loss VGG)
                    # 2e-8 TV (Suavização de ruído)
                    loss = l_mse + (0.006 * l_vgg) + (2e-8 * l_tv)

                scaler.scale(loss).backward()
                scaler.step(opt_gen)
                scaler.update()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            loss_history.append(avg_loss)
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

            # Salvar exemplos e Checkpoints
            if (epoch+1) % 10 == 0 or epoch == 0:
                with torch.no_grad():
                    save_path = OUTPUT_IMGS_DIR / f"sr_epoch_{epoch+1:03d}.png"
                    # Salva: [LowRes (Upscaled bicubic for comparison)] | [Generated SR] | [Real HighRes]
                    lr_upscaled = torch.nn.functional.interpolate(low_res[:4], scale_factor=4)
                    img_grid = torch.cat((lr_upscaled, fake_high_res[:4], high_res[:4]), dim=3)
                    save_image(img_grid, str(save_path))

            # Salvar checkpoint periodicamente (a cada 10 épocas)
            if (epoch + 1) % 10 == 0:
                 torch.save(gen.state_dict(), CHECKPOINTS_DIR / f"sr_generator_finetuned_epoch_{epoch+1}.pth")
            # torch.save(gen.state_dict(), str(MODEL_PATH))

    except KeyboardInterrupt:
        print("Interrompido pelo usuário. Salvando...")

    # Finalização
    torch.save(gen.state_dict(), str(MODEL_PATH))
    
    plt.figure()
    plt.title("Loss Curve SRGAN")
    plt.plot(loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss Combined")
    plt.savefig(str(LOSS_GRAPH_PATH))
    plt.close()
    print(f"Gráfico salvo em {LOSS_GRAPH_PATH}")
    
    print(f"Treinamento finalizado. Modelo salvo em {MODEL_PATH}")

if __name__ == "__main__":
    main()