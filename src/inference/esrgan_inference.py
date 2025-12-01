import torchvision.transforms as transforms
from torchvision.utils import save_image
from pathlib import Path
from PIL import Image
import torch.nn as nn
import torch
import sys
import os

FILE_PATH = Path(__file__).resolve()
# src/inference/esrgan_inference.py -> parents[0]=inference, parents[1]=src, parents[2]=root
PROJECT_ROOT = FILE_PATH.parents[2]
sys.path.append(str(PROJECT_ROOT))

# --- DEFINICAO DAS ARQUITETURAS - IMPORTANTE: (Deve ser IDÊNTICA à usada no treino) ---
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
        # Upsampling: 64 -> 128 -> 256 (4x total)
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

# --- FUNÇÃO DE INFERÊNCIA ---
def upscale_image(image_path, model, device, transform):
    """Carrega uma imagem, faz upscale e retorna o tensor."""
    img = Image.open(image_path).convert("RGB")
    # Transformar para tensor (Normalização não é necessária se treinou só com ToTensor, 
    # mas se usou Normalize no treino, deve usar aqui também. 
    # SRGANs geralmente operam bem apenas com [0, 1])
    img_tensor = transform(img).unsqueeze(0).to(device) # Adiciona dimensão de batch (1, C, H, W)
    with torch.no_grad():
        upscaled_tensor = model(img_tensor)
    return img_tensor, upscaled_tensor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Iniciando Inferência SRGAN Customizada ({device}) ===")

    # Caminhos
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    INPUT_DIR = PROJECT_ROOT / "results" / "inference_dcgan" / "samples"
    OUTPUT_DIR = PROJECT_ROOT / "results" / "inference_srgan"
    MODEL_PATH = CHECKPOINTS_DIR / "sr_generator_finetuned.pth"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregar Modelo
    model = GeneratorSR().to(device)
    
    if MODEL_PATH.exists():
        try:
            model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
            model.eval()
            print(f"Modelo SRGAN carregado de: {MODEL_PATH.name}")
        except Exception as e:
            print(f"ERRO ao carregar pesos: {e}")
            return
    else:
        print(f"ERRO CRÍTICO: Modelo não encontrado em {MODEL_PATH}")
        print("Rode 'src/training/esrgan_finetuning.py' primeiro.")
        return

    # 2. Configurar Entrada
    transform = transforms.ToTensor()

    # Se não houver output da DCGAN, tenta uma pasta genérica de teste
    if not INPUT_DIR.exists() or not any(INPUT_DIR.iterdir()):
        print(f"AVISO: Pasta de entrada da DCGAN vazia ou inexistente: {INPUT_DIR}")
        INPUT_DIR = PROJECT_ROOT / "data" / "test_low_res" # Fallback
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Tentando buscar em fallback: {INPUT_DIR}")

    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("Nenhuma imagem encontrada para processar.")
        print("Dica: Rode 'src/inference/dcgan_inference.py' antes para gerar gatos base.")
        return

    print(f"Processando {len(image_files)} imagens...")

    # 3. Processar cada imagem
    for img_name in image_files:
        input_path = INPUT_DIR / img_name

        lr_tensor, sr_tensor = upscale_image(input_path, model, device, transform)

        # Cria uma comparação visual: [Low Res Upscaled (Bicubic)] | [SRGAN Output]
        # Redimensiona LR via interpolação bicúbica apenas para ficar do mesmo tamanho visual no grid
        lr_resized = torch.nn.functional.interpolate(lr_tensor, scale_factor=4, mode='bicubic')
        
        # Junta lado a lado
        comparison = torch.cat((lr_resized, sr_tensor), dim=3)
        
        save_path = OUTPUT_DIR / f"sr_{img_name}"
        save_image(comparison, str(save_path))
        print(f"Salvo: {img_name} -> sr_{img_name}")

    print("\n--- Concluído ---")
    print(f"Resultados salvos em: {OUTPUT_DIR}")
    print("Nota: As imagens salvas mostram [Esquerda: Bicúbico (Original)] vs [Direita: SRGAN (Seu Modelo)]")

if __name__ == "__main__":
    main()