import torch
from diffusers import StableDiffusionPipeline
import os
from PIL import Image
import shutil

# Assume que o arquivo gif.py está na mesma pasta (mesma lógica do WGAN)
from gif import create_gif

# --- CONFIGURAÇÃO ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Organização de Pastas: Um diretório pai para o experimento, subpasta para frames
EXPERIMENT_NAME = "results_cats_sd"
FRAMES_DIR = os.path.join(EXPERIMENT_NAME, "frames")
FILE_PREFIX = "sd_cat_step_"
NUM_STEPS = 50

# Garante que os diretórios existam
os.makedirs(FRAMES_DIR, exist_ok=True)

# 1. Carregar Pipeline
# Safety checker desabilitado para economizar VRAM e evitar filtros falsos positivos em arte
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16,
    safety_checker=None 
)
pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing() # Essencial para GPUs com < 8GB VRAM

# 2. Prompt
prompt = "a majestic cat portrait, cinematic lighting, highly detailed fur, 8k resolution, oil painting style"
negative_prompt = "blurry, deformed, bad anatomy, low resolution, ugly"

# 3. Função auxiliar para decodificar latents
def decode_tensors(latents):
    # [CORREÇÃO 1] Desanexar do grafo (detach) e garantir o dtype correto (float16)
    # O VAE espera a mesma precisão do modelo carregado.
    latents = latents.detach().to(pipe.vae.dtype)
    
    # Escalar latents (Fator mágico do SD 1.5)
    latents = 1 / 0.18215 * latents
    
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample

    # Normalização manual: [-1, 1] -> [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    
    # Retorna imagem PIL
    return pipe.numpy_to_pil(image)[0]

# 4. Callback para capturar cada passo
def save_step_callback(pipe, step_index, timestep, callback_kwargs):
    # latents agora vêm dentro de um dicionário
    latents = callback_kwargs["latents"]
    
    print(f"Processando passo {step_index+1}/{NUM_STEPS}...")
    img = decode_tensors(latents) # Sua função decode_tensors deve funcionar igual
    
    filename = f"{FRAMES_DIR}/{FILE_PREFIX}{step_index+1:03d}.png"
    img.save(filename)
    return callback_kwargs

print(f"Iniciando geração com Stable Diffusion em: {DEVICE}")

# 5. Inferência
# Definindo seed para reprodutibilidade (prática científica padrão)
generator = torch.Generator(DEVICE).manual_seed(42)

image = pipe(
    prompt, 
    negative_prompt=negative_prompt, 
    num_inference_steps=NUM_STEPS, 
    guidance_scale=7.5,
    generator=generator,
    callback_on_step_end=save_step_callback,
).images[0]

# Salvar resultado final na raiz do experimento
final_path = os.path.join(EXPERIMENT_NAME, "final_result.png")
image.save(final_path)
print(f"Imagem final salva em: {final_path}")

# 6. Gerar GIF
print("Geração concluída. Criando GIF...")

gif_filename = "evolution_process.gif"

create_gif(
    input_folder=FRAMES_DIR, 
    output_gif=gif_filename,
    file_names=FILE_PREFIX, 
    duration=100 
)

source_path = os.path.join(FRAMES_DIR, gif_filename)
destination_path = os.path.join(EXPERIMENT_NAME, gif_filename)

try:
    shutil.move(source_path, destination_path)
    print(f"GIF movido e salvo com sucesso em: {destination_path}")
except FileNotFoundError:
    print(f"Erro: O arquivo {source_path} não foi encontrado. Verifique se o gif.py gerou o arquivo corretamente.")
except Exception as e:
    print(f"Erro ao mover o GIF: {e}")