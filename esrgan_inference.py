"""
Como usar este script:
Certifique-se de que o arquivo de pesos sr_generator_final.pth existe (gerado pelo seu script de finetuning).

Crie uma pasta chamada test_low_res_images no mesmo diretório.

Coloque algumas imagens pequenas (gatos 64x64, ou qualquer imagem pequena) dentro dessa pasta. Você pode pegar algumas geradas pela sua DCGAN.

Rode o script: python srgan_cat_inference.py.

Verifique a pasta test_upscaled_results.
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os

# --- 1. ARQUITETURA (Deve ser IDÊNTICA à usada no treino) ---
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

# --- 2. FUNÇÃO DE INFERÊNCIA ---

def upscale_image(image_path, model, device, transform):
    """Carrega uma imagem, faz upscale e retorna o tensor."""
    # Carregar imagem
    img = Image.open(image_path).convert("RGB")
    
    # Transformar para tensor (Normalização não é necessária se treinou só com ToTensor, 
    # mas se usou Normalize no treino, deve usar aqui também. 
    # SRGANs geralmente operam bem apenas com [0, 1])
    img_tensor = transform(img).unsqueeze(0).to(device) # Adiciona dimensão de batch (1, C, H, W)
    
    with torch.no_grad():
        upscaled_tensor = model(img_tensor)
    
    return upscaled_tensor

def main():
    # --- CONFIGURAÇÕES ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Caminhos (AJUSTE CONFORME NECESSÁRIO)
    _base_dir = "results_cats_dcgan_long_afhq_ram_improved"
    MODEL_PATH = os.path.join(_base_dir, "sr_generator_finetuned_final.pth")
    INPUT_FOLDER = os.path.join(_base_dir, "cat_inference")
    OUTPUT_FOLDER = os.path.join(_base_dir, "cats_upscaled_finetuned")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Carregar Modelo
    print("Carregando modelo SRGAN...")
    model = GeneratorSR().to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval() # Modo de avaliação (trava Batch Norm e Dropout)
        print("Modelo carregado com sucesso!")
    else:
        print(f"ERRO: Arquivo de pesos não encontrado em {MODEL_PATH}")
        print("Você rodou o script de treinamento 'esrgan_cat_upscaler_finetuning.py'?")
        return

    # 2. Preparar Transformação
    transform = transforms.ToTensor()

    # 3. Processar Imagens
    # Se a pasta de entrada não existir, cria uma dummy para o usuário não se perder
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"AVISO: A pasta '{INPUT_FOLDER}' não existia e foi criada.")
        print("Coloque imagens 64x64 lá dentro e rode o script novamente.")
        return

    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Nenhuma imagem encontrada em {INPUT_FOLDER}.")
        return

    print(f"Encontradas {len(image_files)} imagens. Processando...")

    for img_name in image_files:
        input_path = os.path.join(INPUT_FOLDER, img_name)
        
        # Realizar Upscaling
        result_tensor = upscale_image(input_path, model, device, transform)
        
        # Salvar Resultado
        save_path = os.path.join(OUTPUT_FOLDER, f"upscaled_{img_name}")
        save_image(result_tensor, save_path)
        print(f"Salvo: {save_path}")

    print("Inferência finalizada.")

if __name__ == "__main__":
    main()