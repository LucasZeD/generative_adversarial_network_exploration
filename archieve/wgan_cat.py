import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os

from gif import create_gif

# --- CONFIGURAÇÕES AVANÇADAS ---
# WGAN precisa de um Learning Rate menor e Batch Size menor para estabilidade do gradiente
LEARNING_RATE = 1e-4 
BATCH_SIZE = 64
IMAGE_SIZE = 64 
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 500 # WGAN precisa de bastante tempo, mas converge para algo melhor
CRITIC_ITERATIONS = 5 # O Crítico treina 5x para cada 1x do Gerador
LAMBDA_GP = 10 # Peso da penalidade de gradiente (Padrão do paper)

class Critic(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x 3 x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # WGAN não deve usar BatchNorm no crítico. Usamos InstanceNorm.
            self._block(features_d, features_d * 2, 4, 2, 1),      # 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 4x4
            
            # Saída é um score linear (sem Sigmoid!)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0), # 1x1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True), # Instance Norm é melhor para WGAN-GP
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
            
            # Output layer
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1), # 64x64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels), # Gerador ainda pode usar BatchNorm
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# --- A MÁGICA: CÁLCULO DA PENALIDADE DE GRADIENTE ---
def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calcula pontuação das imagens interpoladas
    mixed_scores = critic(interpolated_images)

    # Calcula o gradiente dos scores em relação às imagens
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando WGAN-GP em: {device}")
    
    # Adicionando Data Augmentation (Horizontal Flip) para evitar gatos iguais
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    full_dataset = torchvision.datasets.CIFAR10(root="./data_cifar", train=True, download=True, transform=transform)
    cat_indices = [i for i, (img, label) in enumerate(full_dataset) if label == 3]
    dataset = torch.utils.data.Subset(full_dataset, cat_indices)

    # Worker persistente para evitar crash no Windows
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True 
    )

    gen = Generator(Z_DIM, CHANNELS_IMG, features_g=64).to(device)
    critic = Critic(CHANNELS_IMG, features_d=64).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    # WGAN usa Adam sem momentum inicial (Betas 0.0, 0.9)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
    _output_dir = "results_cats_wgan"
    _file_names = "wgan_cat_epoch_"

    os.makedirs(_output_dir, exist_ok=True)
    gen.train()
    critic.train()

    print("Treinamento Iniciado. O progresso será mais lento por epoch pois o Crítico roda 5x.")

    step = 0
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # 1. Treinar Crítico (Várias vezes)
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = gen(noise)
                
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                
                gp = gradient_penalty(critic, real, fake, device=device)
                
                # A perda na WGAN é D(fake) - D(real) + Penalidade
                # Queremos maximizar a diferença entre real e fake
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # 2. Treinar Gerador (Uma vez)
            output = critic(fake).reshape(-1)
            # O gerador quer maximizar D(fake), então minimizamos -D(fake)
            loss_gen = -torch.mean(output)
            
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        # Logs
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss C: {loss_critic.item():.4f}, Loss G: {loss_gen.item():.4f}")

        with torch.no_grad():
            fake = gen(fixed_noise)
            vutils.save_image(fake, f"{_output_dir}/{_file_names}{epoch+1:03d}.png", normalize=True)

    print("Fim do treinamento WGAN-GP.")
    create_gif(input_folder=_output_dir, output_gif=f"{_file_names}.gif", file_names=_file_names, duration=50)

if __name__ == "__main__":
    main()