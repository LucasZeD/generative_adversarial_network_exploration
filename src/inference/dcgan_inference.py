from pathlib import Path
import torch.nn as nn
import torchvision
import torch
import sys
import os

FILE_PATH = Path(__file__).resolve()
# src/inference/dcgan_inference.py -> parents[0]=inference, parents[1]=src, parents[2]=root
PROJECT_ROOT = FILE_PATH.parents[2]
sys.path.append(str(PROJECT_ROOT))

# --- DEFINICAO DAS ARQUITETURAS - IMPORTANTE: (Deve ser IDÊNTICA à usada no treino) ---
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img):
        super(Generator, self).__init__()
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

def main():
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Iniciando Inferência DCGAN ({device}) ===")

    # Hyperparâmetros (devem ser os mesmos do treino!)
    Z_DIM = 100
    CHANNELS_IMG = 3
    NUM_IMAGES = 64 # Quantidade de imagens a gerar para seleção

    # Caminhos
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    RESULTS_DIR = PROJECT_ROOT / "results" / "inference_dcgan"
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_PATH = CHECKPOINTS_DIR / "dcgan_model_final.pth"

    # 1. Carregar modelo
    gen = Generator(Z_DIM, CHANNELS_IMG).to(device)

    if MODEL_PATH.exists():
        try:
            gen.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
            gen.eval() # IMPORTANTE: Trava o Batch Normalization
            print(f"Sucesso: Modelo carregado de {MODEL_PATH}")
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            return
    else:
        print(f"ERRO CRÍTICO: Arquivo de modelo não encontrado em: {MODEL_PATH}")
        print("Você rodou o script 'src/training/dcgan_training.py'?")
        return
    
    # 2. Carregar os pesos e gerar imagens
    print(f"Gerando {NUM_IMAGES} amostras aleatórias...")
    with torch.no_grad():
        # Gera 64 vetores de ruído diferentes
        noise = torch.randn(NUM_IMAGES, Z_DIM, 1, 1).to(device)
        fake_images = gen(noise)

        # 3. Salvar Resultados

        # 3.A. Salvar Grade (Overview)
        grid_path = RESULTS_DIR / "overview_grid.png"
        torchvision.utils.save_image(fake_images, str(grid_path), normalize=True, nrow=8)
        print(f"Grade salva em: {grid_path}")

        # 3.B. Salvar Imagens Individuais
        # Cria uma subpasta para não poluir
        INDIVIDUAL_DIR = RESULTS_DIR / "samples"
        INDIVIDUAL_DIR.mkdir(exist_ok=True)

        print("Salvando imagens individuais...")
        for i in range(NUM_IMAGES):
            img_name = f"seed_{i:03d}.png"
            save_path = INDIVIDUAL_DIR / img_name
            
            torchvision.utils.save_image(
                fake_images[i], 
                str(save_path), 
                normalize=True
            )
    
    print("\n--- Concluído ---")
    print(f"As imagens foram salvas em: {RESULTS_DIR}")
    print("Dica: Escolha as melhores imagens da pasta 'samples' para testar na SRGAN.")

if __name__ == "__main__":
    main()