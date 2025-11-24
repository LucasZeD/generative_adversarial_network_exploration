import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os

# Tente importar sua função de GIF, se não existir, evita erro
try:
    from gif import create_gif
except ImportError:
    def create_gif(**kwargs):
        print("Módulo gif.py não encontrado. GIF não gerado.")

# --- CONFIGURAÇÕES WGAN-GP ---
BATCH_SIZE = 64 # Reduzido levemente pois WGAN consome mais VRAM no cálculo do gradiente
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 20
LR = 1e-4         # Learning Rate menor é padrão WGAN
BETA1 = 0.0       # Beta1 = 0 é CRUCIAL para WGAN-GP
BETA2 = 0.9
CRITIC_ITERATIONS = 5 # Treina o crítico 5x mais que o gerador
LAMBDA_GP = 10    # Peso da penalidade de gradiente

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self, channels_img):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # WGAN-GP: Não use BatchNorm no Crítico. Use InstanceNorm ou LayerNorm.
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True), 
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            # REMOVIDO: Sigmoid. WGAN usa saída linear (raw score).
        )

    def forward(self, x):
        return self.disc(x).view(-1) # Flatten para [Batch_Size]

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            nn.ConvTranspose2d(channels_noise, 512, kernel_size=4, stride=1, padding=0),
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
            nn.Tanh(), # Saída entre -1 e 1
        )

    def forward(self, x):
        return self.net(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            # Algumas implementações evitam init específico para InstanceNorm, 
            # mas manter normal(1, 0.02) geralmente funciona bem.
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    # 1. Criar imagens interpoladas (mistura de real e fake)
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    # Necessário para calcular gradiente em relação à entrada
    interpolated_images.requires_grad_(True)

    # 2. Calcular scores do crítico para as imagens interpoladas
    mixed_scores = critic(interpolated_images)

    # 3. Calcular gradiente dos scores em relação às imagens
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # 4. Calcular norma do gradiente e penalidade
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1) # L2 norm
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

def main():
    print(f"Device: {DEVICE}")
    print("Iniciando WGAN-GP (High Fidelity Configuration)...")
    
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = torchvision.datasets.MNIST(root=".", transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    gen = Generator(Z_DIM, CHANNELS_IMG).to(DEVICE)
    critic = Critic(CHANNELS_IMG).to(DEVICE)
    
    initialize_weights(gen)
    initialize_weights(critic)

    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_critic = optim.Adam(critic.parameters(), lr=LR, betas=(BETA1, BETA2))

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(DEVICE)
    
    _output_folder = "results_mnist_wgan"
    _file_names = "wgan_"
    os.makedirs(_output_folder, exist_ok=True)

    step = 0
    gen.train()
    critic.train()

    for epoch in range(NUM_EPOCHS):
        for i, (real, _) in enumerate(dataloader):
            real = real.to(DEVICE)
            cur_batch_size = real.shape[0]

            # =============================================================
            # (1) Treino do CRÍTICO
            # Treinamos o crítico mais vezes para ter uma estimativa precisa
            # da Distância Wasserstein
            # =============================================================
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(DEVICE)
                fake = gen(noise)
                
                critic_real = critic(real)
                critic_fake = critic(fake)
                
                gp = gradient_penalty(critic, real, fake, device=DEVICE)
                
                # Loss do Crítico: -(E[real] - E[fake]) + Lambda * GP
                # Queremos maximizar a distância, então minimizamos o negativo dela
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # =============================================================
            # (2) Treino do GERADOR
            # Treina apenas 1 vez a cada CRITIC_ITERATIONS
            # =============================================================
            output = critic(fake)
            # Loss do Gerador: -E[Critic(Fake)]
            # O gerador quer mover as imagens falsas para onde o crítico dá score alto
            loss_gen = -torch.mean(output)
            
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if i % 100 == 0 and i > 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {i}/{len(dataloader)} "
                    f"Loss C: {loss_critic.item():.4f} Loss G: {loss_gen.item():.4f}"
                )

        # Salvar exemplos
        with torch.no_grad():
            fake = gen(fixed_noise)
            vutils.save_image(
                fake,
                f"{_output_folder}/{_file_names}{epoch+1:03d}.png",
                normalize=True,
                nrow=8,
                padding=2
            )
            
    print("Treinamento Finalizado.")
    create_gif(input_folder=_output_folder, output_gif=f"wgan_evolution.gif", file_names=_file_names, duration=500)

if __name__ == "__main__":
    main()