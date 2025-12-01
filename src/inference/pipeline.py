from pathlib import Path
import torch.nn as nn
import torchvision
import torch
import sys
import os

FILE_PATH = Path(__file__).resolve()
# src/inference/pipeline.py -> parents[0]=inference, parents[1]=src, parents[2]=root
PROJECT_ROOT = FILE_PATH.parents[2]
sys.path.append(str(PROJECT_ROOT))

# --- 1. DEFINICAO DAS ARQUITETURAS - IMPORTANTE: (Deve ser IDÊNTICA à usada no treino) ---
# --- DCGAN Generator ---
class DCGANGenerator(nn.Module):
    def __init__(self, z_dim, channels_img):
        super(DCGANGenerator, self).__init__()
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
        )
    def forward(self, x):
        return self.gen(x)

# --- SRGAN Generator ---
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

class SRGANGenerator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=8):
        super(SRGANGenerator, self).__init__()
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

# 2. PIPELINE DE INFERÊNCIA
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Iniciando Pipeline Generativo em Cascata ({device}) ===")

    Z_DIM = 100
    CHANNELS_IMG = 3
    NUM_IMAGES = 64

    # Caminhos
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    OUTPUT_DIR = PROJECT_ROOT / "results" / "pipeline_final"
    DCGAN_WEIGHTS = CHECKPOINTS_DIR / "dcgan_model_final.pth"
    SRGAN_WEIGHTS = CHECKPOINTS_DIR / "sr_generator_finetuned.pth"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Instanciar e Carregar Modelos ---
    # DCGAN
    print(f"[1/4] Carregando DCGAN (Criador)...")
    dcgan = DCGANGenerator(Z_DIM, CHANNELS_IMG).to(device)
    if DCGAN_WEIGHTS.exists():
        try:
            dcgan.load_state_dict(torch.load(str(DCGAN_WEIGHTS), map_location=device))
            dcgan.eval()
        except Exception as e:
            print(f"ERRO ao carregar DCGAN: {e}")
            return
    else:
        print(f"ERRO CRÍTICO: Pesos da DCGAN não encontrados em: {DCGAN_WEIGHTS}")
        print("Certifique-se de ter rodado o treinamento primeiro.")
        return

    # SRGAN
    print(f"[2/4] Carregando SRGAN (Refinador)...")
    srgan = SRGANGenerator().to(device)
    if SRGAN_WEIGHTS.exists():
        try:
            srgan.load_state_dict(torch.load(str(SRGAN_WEIGHTS), map_location=device))
            srgan.eval()
        except Exception as e:
            print(f"ERRO ao carregar SRGAN: {e}")
            return
    else:
        print(f"ERRO CRÍTICO: Pesos da SRGAN não encontrados em: {SRGAN_WEIGHTS}")
        return

    # --- 2. Geração e Refinamento ---
    print(f"[3/4] Gerando {NUM_IMAGES} amostras sintéticas...")
    
    with torch.no_grad():
        # A. Latent Walk -> Low Res Image (DCGAN)
        noise = torch.randn(NUM_IMAGES, Z_DIM, 1, 1).to(device) # ruído aleatório
        low_res_fake = dcgan(noise)
        
        # B. Low Res -> High Res (SRGAN)
        # Importante: SRGAN espera entrada normalizada ou [0,1]? 
        # A DCGAN sai com Tanh (-1 a 1). A SRGAN geralmente espera entrada também normalizada se treinada assim.
        # Como o pipeline é contínuo, passamos o tensor direto.
        high_res_fake = srgan(low_res_fake)

    # --- 3. Salvar Resultados ---
    print(f"[4/4] Salvando resultados em '{OUTPUT_DIR}'...")

    # Salva Grid Low Res (64x64)
    torchvision.utils.save_image(
        low_res_fake, 
        str(OUTPUT_DIR / "stage1_dcgan_64x64.png"), 
        normalize=True, 
        nrow=4
    )

    # Salva Grid High Res (256x256)
    torchvision.utils.save_image(
        high_res_fake, 
        str(OUTPUT_DIR / "stage2_srgan_256x256.png"), 
        normalize=True, 
        nrow=4
    )

    # Salva pares individuais para comparação detalhada
    print("Gerando comparações lado a lado...")
    COMPARISON_DIR = OUTPUT_DIR / "comparisons"
    COMPARISON_DIR.mkdir(exist_ok=True)

    for i in range(NUM_IMAGES):
        # Cria uma imagem combinada: [Low Res Redimensionada] | [High Res Original]
        lr_upscaled = torch.nn.functional.interpolate(low_res_fake[i].unsqueeze(0), scale_factor=4, mode='bicubic')
        comparison = torch.cat((lr_upscaled, high_res_fake[i].unsqueeze(0)), dim=3)
        
        torchvision.utils.save_image(
            comparison,
            str(COMPARISON_DIR / f"final_result_{i+1:02d}.png"),
            normalize=True
        )

    print("\n--- Concluído ---")
    print(f"1. Visão Geral 64px:  {OUTPUT_DIR / 'stage1_dcgan_64x64.png'}")
    print(f"2. Visão Geral 256px: {OUTPUT_DIR / 'stage2_srgan_256x256.png'}")
    print(f"3. Detalhes: Verifique a pasta '{COMPARISON_DIR}'")

if __name__ == "__main__":
    main()