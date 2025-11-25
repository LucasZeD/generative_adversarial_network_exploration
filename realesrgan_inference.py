"""
Real-ESRGAN (2021)
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git

4. Como explicar isso na apresentação (Argumento Acadêmico)
    Para não parecer que você apenas "rodou um script pronto", você deve justificar a escolha arquitetural.
    Aqui está o roteiro técnico:
        O Problema da DCGAN: "A DCGAN é excelente para aprender a distribuição de probabilidade das imagens (formas, cores), mas sofre para gerar altas frequências (detalhes finos de textura) devido à limitação de VRAM e instabilidade de treino em altas resoluções."
    A Solução Híbrida:
        "Utilizamos uma abordagem em dois estágios (Two-Stage Generation).
        O Estágio 1 (DCGAN) cria a estrutura semântica e global do gato.
        O Estágio 2 (Real-ESRGAN) atua como um refinador de textura."
    Por que não Fine-tuning?:
        "Optou-se por usar a Real-ESRGAN em modo de inferência Zero-Shot pois ela foi treinada com degradações sintéticas complexas que simulam os artefatos gerados por redes neurais, tornando-a mais robusta para limpar o ruído da DCGAN do que um modelo treinado do zero com downsampling bicúbico simples."

5. Resultado Esperado
    Input: Gato 64x64 (um pouco borrado, pixelado).
    Output: Gato 256x256.
        O olho ficará nítido.
        A massade cores do corpo ganhará textura de fios de pelo.
        Artefatos de "tabuleiro de xadrez" (checkerboard artifacts) da DCGAN tendem a ser suavizados.
"""

import torch
import huggingface_hub
import os
import shutil

def smart_cached_download(url_or_id, **kwargs):
    """
    Wraps hf_hub_download to handle legacy calls from sberbank-ai/Real-ESRGAN.
    1. Parses full URLs into repo_id.
    2. Maps 'force_filename' to 'filename'.
    3. Copies the file from the HF cache to the local folder expected by RealESRGAN.
    """
    # 1. Handle URL vs Repo ID mismatch
    if "huggingface.co" in url_or_id and "Real-ESRGAN" in url_or_id:
        repo_id = "sberbank-ai/Real-ESRGAN"
    else:
        repo_id = url_or_id

    # 2. Handle deprecated arguments
    filename = kwargs.pop('force_filename', None)
    if not filename:
        filename = url_or_id.split('/')[-1] if "huggingface.co" in url_or_id else "RealESRGAN_x4.pth"

    # Capture the target directory RealESRGAN wants to use (e.g., "weights/")
    cache_dir_arg = kwargs.pop('cache_dir', None)
    kwargs.pop('use_auth_token', None)

    # 3. Download to Hugging Face Cache first
    print(f"DEBUG: Downloading {filename} from {repo_id}...")
    cached_path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)

    # 4. Copy to the actual location RealESRGAN expects
    # RealESRGAN expects the file to be at: os.path.join(cache_dir, filename)
    if cache_dir_arg:
        destination_path = os.path.join(cache_dir_arg, filename)
        os.makedirs(cache_dir_arg, exist_ok=True)
        
        # Only copy if we need to (or simply overwrite to ensure it's correct)
        print(f"DEBUG: Copying from cache to {destination_path}")
        shutil.copy2(cached_path, destination_path)
        return destination_path
    
    return cached_path

huggingface_hub.cached_download = smart_cached_download
from RealESRGAN import RealESRGAN
from PIL import Image
import numpy as np

def upscale_images():
    # Configurações
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Carrega o modelo Real-ESRGAN pré-treinado (escala 4x)
    # O download dos pesos será feito automaticamente na primeira execução
    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)

    # Pastas
    _base_folder = "results_cats_dcgan_long_afhq_ram_improved"
    input_folder = os.path.join(_base_folder, "cat_inference")
    output_folder = os.path.join(_base_folder, "cats_upscaled")
    os.makedirs(output_folder, exist_ok=True)

    # Processamento
    print(f"Iniciando Upscaling de {input_folder} para {output_folder}...")
    
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg'))]
    
    if not image_files:
        print(f"Nenhuma imagem encontrada em {input_folder}. Rode o script de inferência anterior primeiro.")
        return

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        
        # Abrir imagem
        image = Image.open(img_path).convert('RGB')
        
        # Predição (Upscaling)
        try:
            sr_image = model.predict(image)
            
            # Salvar
            save_path = os.path.join(output_folder, f"sr_{img_name}")
            sr_image.save(save_path)
            print(f"Upscaled: {img_name} -> 256x256")
        except Exception as e:
            print(f"Erro ao processar {img_name}: {e}")

    print("Finalizado. Compare as imagens nas pastas.")

if __name__ == "__main__":
    upscale_images()