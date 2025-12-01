from diffusers import StableDiffusionPipeline
from pathlib import Path
import torch
import sys
import os

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
sys.path.append(str(PROJECT_ROOT))

def run_stable_diffusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stable Diffusion Inference ({device}) ===")

    if device.type == 'cpu':
        print("AVISO CRÍTICO: Rodar SD em CPU é extremamente lento (minutos por imagem).")

    # Configurações
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    OUTPUT_DIR = PROJECT_ROOT / "results" / "inference_stable_diffusion"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prompt Engineering (O "Código Latente" agora é Texto)
    PROMPT = "a majestic cat portrait, cinematic lighting, highly detailed fur, 8k resolution, oil painting style"
    NEGATIVE_PROMPT = "blurry, deformed, bad anatomy, low resolution, ugly, extra limbs"
    
    NUM_IMAGES = 4
    STEPS = 150         # Quantidade de passos de difusão (mais = melhor qualidade, mas mais lento)
    GUIDANCE = 7.5      # O quanto a IA deve obedecer fielmente ao prompt

    # 1. Carregar o Pipelineel
    print(f"Baixando/Carregando modelo: {MODEL_ID}...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
        pipe = pipe.to(device)
        
        # Otimização de Memória (Slice Attention) - Ajuda se tiver < 8GB VRAM
        if device.type == 'cuda':
            pipe.enable_attention_slicing()
            
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        print("Verifique sua conexão e se 'diffusers' está instalado.")
        return

    # 2. Geração
    print(f"Gerando {NUM_IMAGES} imagens de gatos...")
    
    for i in range(NUM_IMAGES):
        print(f"Gerando {i+1}/{NUM_IMAGES}...")
        # A mágica acontece aqui
        image = pipe(
            PROMPT, 
            negative_prompt=NEGATIVE_PROMPT, 
            num_inference_steps=STEPS, 
            guidance_scale=GUIDANCE
        ).images[0]
        
        # Salvar
        save_path = OUTPUT_DIR / f"sd_cat_{i+1:03d}.png"
        image.save(save_path)
        print(f"Salvo: {save_path}")

    print("\n--- Concluído ---")
    print(f"Resultados em: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_stable_diffusion()