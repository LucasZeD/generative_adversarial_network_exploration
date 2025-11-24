import torch
from diffusers import StableDiffusionPipeline

# --- CONFIGURAÇÃO ---
# Usamos float16 para economizar VRAM e ganhar velocidade (quase sem perda de qualidade)
device = "cuda"
model_id = "runwayml/stable-diffusion-v1-5" # O clássico, leve e eficiente

# 1. Carregar o Pipeline (VAE + U-Net + Scheduler + Text Encoder)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# 2. Otimização para RTX (Attention Slicing economiza VRAM)
pipe.enable_attention_slicing()

# 3. O Prompt (A "Mágica")
# Aqui pedimos o gato que a WGAN sofreu para fazer
prompt = "a majestic cat portrait, cinematic lighting, highly detailed fur, 8k resolution, oil painting style"
negative_prompt = "blurry, deformed, bad anatomy, low resolution"

print("Gerando imagem com Stable Diffusion...")

# 4. Inferência (Denoising Loop)
# num_inference_steps=50: O "Escultor" vai dar 50 marteladas para limpar o ruído
image = pipe(
    prompt, 
    negative_prompt=negative_prompt, 
    num_inference_steps=50, 
    guidance_scale=7.5 # O quanto a rede deve obedecer o texto (vs criatividade livre)
).images[0]

# 5. Salvar
image.save("sd_cat_masterpiece.png")
print("Imagem salva: sd_cat_masterpiece.png")