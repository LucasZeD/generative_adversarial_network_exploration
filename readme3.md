# Generative Pipeline: DCGAN + SRGAN (Cat Domain)

Este projeto implementa um pipeline generativo em cascata para síntese de imagens de gatos. O sistema utiliza uma abordagem em dois estágios:
1.  **Geração Semântica (Low-Res):** Uma DCGAN treinada do zero gera a estrutura global da imagem (64x64).
2.  **Refinamento de Textura (Super-Resolution):** Comparação entre um modelo de mercado (Real-ESRGAN) e uma SRGAN customizada treinada especificamente no domínio felino para realizar *upscaling* (4x) para 256x256.

---

## 📂 Estrutura do Projeto e Fluxo de Dados

### 1. Treinamento Base (DCGAN)
**Script:** `dcgan_cat_long_afhq_ram.py`
* **Descrição:** Treinamento da *Deep Convolutional GAN* do zero.
* **Input:** Dataset AFHQ Cats (Redimensionado para 64x64).
* **Output (Pasta `epochs`):**
    * Checkpoints visuais durante o treinamento.
    * **Modelo Final:** `generator_final.pth` (O "cérebro" gerador de baixa resolução).

### 2. Inferência Latente
**Script:** `dcgan_cat_inference.py`
* **Descrição:** Amostragem do espaço latente utilizando o gerador treinado.
* **Input:** Vetor de Ruído Gaussiano $z \sim N(0, 1)$ (dimensão 100).
* **Output (Pasta `cat_inference`):**
    * Imagens 64x64 geradas sinteticamente.

### 3. Benchmark de Upscaling (Baseline)
**Script:** `realesrgan_cat_upscaler.py`
* **Descrição:** Aplicação *Zero-Shot* do modelo Real-ESRGAN (x4plus) pré-treinado em dados genéricos. Serve como base de comparação.
* **Input:** Imagens 64x64 da pasta `cat_inference`.
* **Output (Pasta `cats_upscaled`):**
    * Imagens 256x256 refinadas por modelo genérico.

### 4. Treinamento do Upscaler (Domain Adaptation)
**Script:** `esrgan_cat_upscaler_finetuning.py`
* **Descrição:** Treinamento de uma SRGAN (Super-Resolution GAN) utilizando *Perceptual Loss* (VGG19) especificamente no dataset de gatos.
* **Input:** Dataset AFHQ (Pares de imagem: Low-Res 64x64 $\to$ High-Res 256x256 gerados on-the-fly).
* **Output (Pasta `epochs_finetune`):**
    * Resultados de validação durante o treino.
    * **Modelo Final:** `sr_generator_final.pth` (Especialista em textura de gatos).

### 5. Inferência Final (Custom SRGAN)
**Script:** `srgan_cat_inference.py`
* **Descrição:** Aplicação do modelo SRGAN customizado nas imagens geradas pela DCGAN.
* **Input:** Imagens 64x64 da pasta `cat_inference`.
* **Output (Pasta `cats_upscaled_finetuned`):**
    * Imagens 256x256 finais com texturas restauradas pelo modelo especialista.

---

## 🚀 Como Executar (Pipeline Completo)

Para rodar o fluxo completo de ponta a ponta (geração + upscale), execute o script de automação:

```bash
python image_pipeline.py