from pathlib import Path
import huggingface_hub
from PIL import Image
import numpy as np
import shutil
import torch
import sys
import os

FILE_PATH = Path(__file__).resolve()
# src/inference/realesrgan_inference.py -> parents[0]=inference, parents[1]=src, parents[2]=root
PROJECT_ROOT = FILE_PATH.parents[2]
sys.path.append(str(PROJECT_ROOT))

# A biblioteca sberbank-ai/Real-ESRGAN usa chamadas depreciadas. 
# Esta função intercepta o download e o corrige para versões modernas.

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

    cache_dir_arg = kwargs.pop('cache_dir', None)
    kwargs.pop('use_auth_token', None)

    # 3. Download to Hugging Face Cache first
    try:
        cached_path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    except Exception as e:
        # Fallback para tentar baixar sem argumentos extras que podem quebrar
        cached_path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)

    # 4. Copy to the actual location RealESRGAN expects
    # RealESRGAN expects the file to be at: os.path.join(cache_dir, filename)
    if cache_dir_arg:
        destination_path = os.path.join(cache_dir_arg, filename)
        os.makedirs(cache_dir_arg, exist_ok=True)
        if not os.path.exists(destination_path):
            shutil.copy2(cached_path, destination_path)
        return destination_path
    
    return cached_path

huggingface_hub.cached_download = smart_cached_download
from RealESRGAN import RealESRGAN

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Iniciando Real-ESRGAN (Baseline) em: {device} ===")

    # Caminhos
    INPUT_DIR = PROJECT_ROOT / "results" / "inference_dcgan" / "samples"
    OUTPUT_DIR = PROJECT_ROOT / "results" / "inference_realesrgan"
    WEIGHTS_DIR = PROJECT_ROOT / "checkpoints" / "realesrgan_weights"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Carrega o modelo Real-ESRGAN pré-treinado (escala 4x)
    # O download dos pesos será feito automaticamente na primeira execução
    try:
        model = RealESRGAN(device, scale=4)
        # Salva os pesos na pasta checkpoints para organização
        weights_path = WEIGHTS_DIR / "RealESRGAN_x4.pth"
        model.load_weights(str(weights_path), download=True)
    except Exception as e:
        print(f"ERRO CRÍTICO ao carregar Real-ESRGAN: {e}")
        print("Verifique se instalou: pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git")
        return

    # Verificar Entrada
    if not INPUT_DIR.exists() or not any(INPUT_DIR.iterdir()):
        print(f"AVISO: Nenhuma imagem encontrada em {INPUT_DIR}")
        print("Dica: Rode 'src/inference/dcgan_inference.py' primeiro.")
        return
    
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Processando {len(image_files)} imagens...")

    for img_name in image_files:
        img_path = INPUT_DIR / img_name
        
        # Abrir imagem
        image = Image.open(img_path).convert('RGB')
        
        try:
            # Predição (Upscaling)
            sr_image = model.predict(image)
            
            # Criar Comparação: [Original Resized Bicubic] | [Real-ESRGAN]
            # Redimensiona a original apenas para ficar do mesmo tamanho visual
            w, h = sr_image.size
            original_resized = image.resize((w, h), Image.BICUBIC)
            
            # Cria nova imagem combinada
            comparison = Image.new('RGB', (w * 2, h))
            comparison.paste(original_resized, (0, 0))
            comparison.paste(sr_image, (w, 0))
            
            # Salvar
            save_path = OUTPUT_DIR / f"realesrgan_{img_name}"
            comparison.save(save_path)
            print(f"Salvo: realesrgan_{img_name}")
            
        except Exception as e:
            print(f"Erro ao processar {img_name}: {e}")

    print("\n--- Concluído ---")
    print(f"Resultados salvos em: {OUTPUT_DIR}")
    print("Nota: As imagens mostram [Esquerda: Bicúbico] vs [Direita: Real-ESRGAN]")

if __name__ == "__main__":
    main()