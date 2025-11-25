import torch
import torchvision
import os
import torch.nn as nn

# --- IMPORTANTE: Copie a classe Generator EXATAMENTE como no treino ---
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
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

def generate_cherry_picking():
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Z_DIM = 100
    CHANNELS_IMG = 3
    
    # Caminho do modelo salvo (ajuste conforme o nome da sua pasta)
    _base_folder = "results"
    _model_name = "dcgan_model_final.pth"
    _output_folder = "dcgan_inference"
    MODEL_PATH = os.path.join(_base_folder, _model_name)
    OUTPUT_FOLDER = os.path.join(_base_folder, _output_folder)
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Carregar a arquitetura
    gen = Generator(Z_DIM, CHANNELS_IMG).to(device)
    
    # 2. Carregar os pesos (o cérebro treinado)
    if os.path.exists(MODEL_PATH):
        gen.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Modelo carregado com sucesso!")
    else:
        print(f"ERRO: Arquivo {MODEL_PATH} não encontrado. Você salvou o .pth?")
        return

    # Colocar em modo de avaliação (importante para Batch Norm)
    gen.eval()

    print("Gerando 64 gatos para seleção...")
    
    with torch.no_grad():
        # Gera 64 vetores de ruído diferentes
        noise = torch.randn(64, Z_DIM, 1, 1).to(device)
        fake_images = gen(noise)

        # Salvar UMA grade com todos (para visão geral)
        torchvision.utils.save_image(fake_images, f"{OUTPUT_FOLDER}/all_grid.png", normalize=True, nrow=8)

        # Salvar CADA imagem individualmente (para você escolher as melhores)
        for i in range(64):
            # Salva individualmente: cat_001.png, cat_002.png...
            torchvision.utils.save_image(
                fake_images[i], 
                f"{OUTPUT_FOLDER}/cat_{i+1:03d}.png", 
                normalize=True
            )
            
    print(f"Pronto! Verifique a pasta '{OUTPUT_FOLDER}'.")
    print("Agora escolha manualmente as 8 imagens que parecem mais reais.")

if __name__ == "__main__":
    generate_cherry_picking()