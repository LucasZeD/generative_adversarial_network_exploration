## Dataset
https://www.kaggle.com/datasets/dimensi0n/afhq-512

### Configuração de Ambiente
1. Criar um Ambiente Virtual 
```bash
python -m venv .venv
```

2. Ativar o Ambiente
```bash
# MacOs/linus
source .venv/bin/activate

#windws (cmd)
.\.venv\Scripts\activate.bat
```

3. Instalar dependências
torchvision para cuda
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
upscaler Real Enhanced Super Resolution GAN
```bash
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
```
outras dependencias
```bash
pip install -r requirements.txt
```

### Gerando .zip para envio
```bash
git archive --format=zip --output=project-main.zip <branch-name>
```