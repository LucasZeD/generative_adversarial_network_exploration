"""
1. Treinamento DCGAN (Geração Criativa)
    Ação: Baixar dataset AFHQ -> Redimensionar tudo para $64 \times 64$ -> Treinar DCGAN.
    Output: Arquivo generator_final.pth (O "Cérebro" que sabe imaginar gatos pequenos)

2. Treinamento SRGAN (O "Fine-tuning" do Upscaler)
    Ação: Usar o mesmo dataset AFHQ original ($256 \times 256$ ou $512 \times 512$).
    O Truque: O código pega a imagem HD real, diminui para $64 \times 64$ "na hora" (on-the-fly) e diz para a rede:
        "Aprenda a transformar essa miniatura de volta na original."
    Output: Arquivo sr_generator_final.pth (O "Cérebro" que sabe restaurar texturas de gatos).

3. O Passo Final: Inferência Integrada (Pipeline Completo)
Aqui você responde sua dúvida ("inferencia ?"). O passo final é conectar os dois cérebros.
Você não precisa salvar as imagens no meio do caminho se não quiser, pode passar os dados direto na memória.
    Fluxo da Inferência Final:
        Gerar Ruído Aleatório (Latent Vector).'
        Passar pelo DCGAN $\to$ Sai Gato $64 \times 64$.
        Pegar esse Gato $64 \times 64$ e passar direto pelo SRGAN.
        Resultado Final: Gato $256 \times 256$ (Criado do zero e refinado).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast  # OTIMIZAÇÃO DE VELOCIDADE
import multiprocessing

# --- 1. ARQUITETURA DO GERADOR (Inalterada) ---
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

# --- 2. LOSS FUNCTIONS (CORRIGIDO PARA REMOVER RUÍDO) ---

class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Carrega VGG até a camada 35 (features de textura)
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:35].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        
        # CORREÇÃO CRÍTICA: Normalização da ImageNet
        # Sem isso, a VGG "vê" ruído onde não existe
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]).to(device)
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        # Normaliza antes de passar na VGG
        vgg_input = self.normalize(input)
        vgg_target = self.normalize(target)
        return self.mse(self.vgg(vgg_input), self.vgg(vgg_target))

class TVLoss(nn.Module):
    """Total Variation Loss - O 'Desembaçador' de Ruído"""
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

# --- 3. DATASET ---
class CatSRDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.data = datasets.ImageFolder(root=root_dir)
        self.high_res_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.low_res_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, _ = self.data[index]
        high_res = self.high_res_transform(img)
        low_res = self.low_res_transform(img)
        return low_res, high_res

def main():
    # OTIMIZAÇÃO 1: cudnn Benchmark
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando SRGAN OTIMIZADA em: {device}")

    # Configurações
    LR = 1e-4
    BATCH_SIZE = 16 
    EPOCHS = 100
    
    # OTIMIZAÇÃO 2: Workers Inteligentes
    num_workers = min(8, os.cpu_count()) if os.name != 'nt' else 0
    print(f"Workers: {num_workers}")
    
    dataset_path = "./data_cats/train"
    base_dir = "results_cats_dcgan_long_afhq_ram_improved"
    output_dir = os.path.join(base_dir, "epochs_finetune_clean")
    os.makedirs(output_dir, exist_ok=True)

    dataset = CatSRDataset(root_dir=dataset_path)
    
    # OTIMIZAÇÃO 3: Persistent Workers e Pin Memory
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        pin_memory=True, num_workers=num_workers,
                        persistent_workers=(num_workers > 0))

    gen = GeneratorSR().to(device)
    opt_gen = optim.Adam(gen.parameters(), lr=LR)
    
    # Losses
    mse_loss = nn.MSELoss()
    vgg_loss = VGGLoss(device).to(device)
    tv_loss = TVLoss(tv_loss_weight=1e-5).to(device) # Peso baixo, apenas para limpar ruído

    # OTIMIZAÇÃO 4: Mixed Precision Scaler
    scaler = GradScaler()
    
    loss_history = []

    print("Iniciando Treinamento Rápido e Limpo...")

    for epoch in range(EPOCHS):
        epoch_loss = 0
        
        for idx, (low_res, high_res) in enumerate(loader):
            low_res = low_res.to(device, non_blocking=True)
            high_res = high_res.to(device, non_blocking=True)

            opt_gen.zero_grad()

            # OTIMIZAÇÃO 5: Autocast Context (Roda em FP16 quando possível)
            with autocast():
                fake_high_res = gen(low_res)
                
                # CÁLCULO DE PERDA OTIMIZADO PARA QUALIDADE
                l_mse = mse_loss(fake_high_res, high_res)
                l_vgg = vgg_loss(fake_high_res, high_res)
                l_tv = tv_loss(fake_high_res) # Penaliza ruído
                
                # Equilíbrio Fino:
                # MSE: 1.0 (Base estrutural)
                # VGG: 0.006 (Textura realista)
                # TV:  2e-8  (Limpeza de ruído - evita granulação)
                loss = l_mse + (0.006 * l_vgg) + (2e-8 * l_tv)

            # Backward Otimizado
            scaler.scale(loss).backward()
            scaler.step(opt_gen)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

        if (epoch+1) % 10 == 0:
            save_image(fake_high_res[:8], f"{output_dir}/sr_clean_epoch_{epoch+1}.png")
            # Salvar checkpoint
            torch.save(gen.state_dict(), f"{base_dir}/sr_generator_clean.pth")

    # Gráfico Final
    plt.figure()
    plt.plot(loss_history)
    plt.title("Loss Treinamento Otimizado")
    plt.savefig(f"{base_dir}/loss_optimized.png")
    print("Concluído.")

if __name__ == "__main__":
    main()