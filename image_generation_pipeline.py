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


Workflow Profissional:
    "Professor, eu não apenas gerei imagens. Eu criei um Pipeline Generativo em Cascata."
    Estágio 1 (Criação):
        DCGAN gera a geometria e semântica global (forma do gato) em espaço latente comprimido (64x64).
    Estágio 2 (Refinamento):
        SRGAN (treinada especificamente no domínio felino) alucina os detalhes de alta frequência (pelos, brilho) fazendo o mapeamento $64 \to 256$.
    Por que treinar? (Sua pergunta inicial):
        "Modelos prontos usam interpolação bicúbica durante o treino.
        A minha SRGAN aprendeu a desfazer os artefatos específicos da DCGAN, agindo também como um Denoising Autoencoder, não apenas um Upscaler."
"""

import torch
import torchvision
import os
# Importe as classes que definimos nos arquivos anteriores
# Supondo que você salvou as classes em arquivos separados ou copiará para cá
from Trabalho.gan_exploration.dcgan_training import Generator as DCGANGenerator
from esrgan_cat_upscaler_finetuning import GeneratorSR as SRGANGenerator

def run_full_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando Pipeline Completo em: {device}")

    # 1. Carregar DCGAN (O Criador)
    dcgan = DCGANGenerator(z_dim=100, channels_img=3).to(device)
    dcgan.load_state_dict(torch.load("results_cats_dcgan_improved/generator_final.pth", map_location=device))
    dcgan.eval()

    # 2. Carregar SRGAN (O Refinador)
    srgan = SRGANGenerator().to(device)
    srgan.load_state_dict(torch.load("results_sr_cats/sr_generator_final.pth", map_location=device))
    srgan.eval()

    # 3. Gerar Imagens
    num_images = 16
    noise = torch.randn(num_images, 100, 1, 1).to(device)

    print("Gerando e Ampliando...")
    with torch.no_grad():
        # Estágio 1: Low Res (64x64)
        low_res_fake = dcgan(noise)
        
        # Estágio 2: Super Res (256x256)
        # O SRGAN pega a saída da DCGAN como entrada
        high_res_fake = srgan(low_res_fake)

    # 4. Salvar Resultados para Comparação
    os.makedirs("final_results", exist_ok=True)
    
    # Salva o Low Res
    torchvision.utils.save_image(low_res_fake, "final_results/stage1_low_res.png", normalize=True, nrow=4)
    
    # Salva o High Res
    torchvision.utils.save_image(high_res_fake, "final_results/stage2_high_res.png", normalize=True, nrow=4)
    
    print("Pipeline concluído! Verifique a pasta 'final_results'.")
    print("Compare stage1_low_res.png com stage2_high_res.png para ver a diferença.")

if __name__ == "__main__":
    run_full_pipeline()