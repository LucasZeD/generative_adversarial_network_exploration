# Exploration of Image Generation Algorithms Based on Generative Adversarial Network (GANs)

Este projeto implementa um pipeline para treinamento e inferencia de modelos baseados em Redes Adversariais Generativas (GANs). 
1. **Geração Semântica (DCGAN)**: Gera a estrutura global da imagem em baixa resolução ($64 \times 64$).
2. **Refinamento de Textura (SRGAN)**: Realiza upscaling (4x) para ($256 \times 256$) recuperando detalhes de alta frequência (pelos, texturas).

## Dataset
O projeto utiliza o dataset `Kaggle - Animal Faces High Quality Dataset` ([AFHQ](https://www.kaggle.com/datasets/dimensi0n/afhq-512)), filtrado para a classe de gatos.

As imagens devem ser organizadas na pasta `./data/train` para o carregamento correto pelos scripts.

## Configuração de Ambiente e Execução

**1. Criar e ativar o Ambiente Virtual**
```bash
# Criar ambiente virtual
python -m venv .venv

# MacOs/linus
source .venv/bin/activate
# Windws (cmd)
.\.venv\Scripts\activate.bat

# Instalar PyTorch com suporte a CUDA (Ajuste conforme sua versão do CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Instalar Real-ESRGAN (upscaler baseline) e dependências
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git

# Demais dependencias
pip install -r requirements.txt
```

**2. Configuração do Dataset**
Baixar o dataset desejado e adicionar as imagens na pasta `./data/train` para processamento pelos scripts

## Execução do Pipeline
O fluxo de trabalho deve ser executado na ordem abaixo apra gerar os modelos (.pth) necessários.

- Acompanhar uso da GPU
```bash
nvidia-smi -l 1
```

**Passo 1:** Treinamento DCGAN - Gerador Base
Treina a rede convolucional para aprender a distribuição dos gatos em ($64 \times 64$).
- Input: Imagens AFHQ redimensionadas. (`data/train`)
- Ouput: `results/dcgan_model_final.pth`
```bash 
python src/training/dcgan_training.py
```

**Passo 2:** Fine-tuning da SRGAN - Upscaler Base
Treina o modelo de Super-Resolução usando VGG19 para aprender texturas especifícas para os gatos, limpando artefatos da geração base.
- Output: `checkpoints/sr_generator_finetuned.pth`
```bash
python src/training/esrgan_finetuning.py
```

**Passo 3:** Inferência e Comparação
Gere imagens e compare o upscaling customizado contra o baseline de mercado.
```bash
# A. Gerar imagens 64x64
python src/inference/dcgan_inference.py
# B. Upscale Genérico (Real-ESRGAN - Baseline)
python src/inference/realesrgan_inference.py
# C. Upscale 'Especializado' (Finetuned-SRGAN)
python src/inference/esrgan_inference.py
```

**Passo 4:** Pipeline Completo Automatizado
Executa o fluxo completo (Ruído $\to$ DCGAN $\to$ SRGAN $\to$ Imagem Final).
```bash
python src/inference/pipeline.py
```

## Definição de HyperParametros

| Modelo | Componente   | Configuração | Detalhes                                                                                                  |
|--------|--------------|--------------|-----------------------------------------------------------------------------------------------------------|
| DCGAN  | Geral        | Epochs: 700  | Batch Size: 128, Img Size: 64x64                                                                          |
|        | Otimizador   | Adam         | LR Gerador: $0.0002$, LR Discriminador: $0.00005$, $\beta_1=0.5$                                          |
|        | Estabilidade |              | Spectral Normalization (Discriminador), Label Smoothing (0.9), Noise Injection (decai ao longo do tempo). |
| SRGAN  | Geral        | Epochs: 100  | Batch Size: 128, Img Size: 64x64                                                                          |
|        | Loss         | Híbrida      | MSE (Estrutura) + VGG19 (Textura/Perceptual) + TV Loss (Denoising).                                       |
|        | Escala       | 4x           | Input: 64x64 $\to$ Output: 256x256                                                                        |

## Estrutura do Projeto e Fluxo de Dados

1. Estrutura de Pastas
```plaintext
generative_adversarial_network_exploration/
│
├── data/                       # Dados brutos.
│   └── train/                  # Imagens do AFHQ (gatos)
│
├── src/
│   ├── __init__.py
│   ├── training/               # Scripts de treinamento
│   │   ├── dcgan_training.py
│   │   └── esrgan_finetuning.py
│   │
│   ├── inference/              # Scripts de inferência/geração
│   │   ├── dcgan_inference.py
│   │   ├── esrgan_inference.py
│   │   ├── realesrgan_inference.py
│   │   └── pipeline.py         # Renomeado de image_inference_pipeline.py
│   │
│   └── utils/                  # Funções auxiliares
│       └── gif.py
│
├── checkpoints/                # (.pth) Pesos treinados e modelos finais
│   ├── dcgan_model_final.pth
│   ├── sr_generator_finetuned.pth
│   └── ...
│
├── results/                    # Saídas visuais (imagens geradas/logs)
│   ├── training_logs/          # Gifs e imagens de épocas
│   ├── inference_dcgan/
│   ├── inference_srgan/
│   └── pipeline_final/         # Saída do Pipeline completo
|
├── results/                    # best saved results
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

2. Estrutura de Saída
- `checkpoints/`: Modelos treidanos (.pth)
- `results/training_logs`: Gifs de evolução do treinamento
- `results/pipeline_final`: Resultado final da integração dos modelos.

## Comparativo de Arquiteturas

| Característica           | GAN                                                                                          | Stable Diffusion                                                             |
|--------------------------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| Velocidade de Inferência | Instantânea (ms). Gera em um passo.                                                          | Lenta (segundos). Requer 20-50 passos iterativos.                            |
| Controle                 | Espaço Latente Matemático ($z$). Difícil de controlar semanticamente sem técnicas avançadas. | Prompt de Texto. Controle semântico nativo ("gato azul").                    |
| Estabilidade de Treino   | Baixa. Sofre de Mode Collapse (gera sempre o mesmo gato) e oscilação.                        | Alta. A função de perda é mais simples (MSE no ruído), não tem "adversário". |
| Resolução Nativa         | Difícil escalar. Geralmente baixa (64x64, 128x128).                                          | Alta (512x512, 1024x1024) por padrão.                                        |
| Uso Principal            | Upscaling (SRGAN), Transferência de Estilo em tempo real.                                    | Geração de Arte, Edição de Imagem, Inpainting.                               |