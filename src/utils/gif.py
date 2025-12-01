from PIL import Image
import glob
import os

def create_gif(input_folder, output_gif, file_names="epoch_", duration=100):
    """
    Gera um GIF a partir de imagens em uma pasta.
    
    Args:
        input_folder (str): Caminho da pasta onde estão as imagens (ex: results/logs).
        output_gif (str): Caminho COMPLETO do arquivo de saída (ex: results/video.gif).
        file_names (str): Prefixo dos arquivos a serem buscados.
        duration (int): Duração de cada frame em ms.
    """
    # 1. Buscar todas as imagens que começam com o padrão especificado
    search_pattern = os.path.join(input_folder, file_names + "*.png")
    filenames = sorted(glob.glob(search_pattern))

    if not filenames:
        print(f"AVISO (GIF): Nenhuma imagem encontrada em '{search_pattern}'.")
        return

    print(f"GIF: Encontradas {len(filenames)} imagens. Processando...")

    # 2. Carregar imagens na memória
    frames = []
    for f in filenames:
        try:
            frames.append(Image.open(f))
        except Exception as e:
            print(f"Erro ao ler {f}: {e}")

    if not frames:
        return

    # 3. Salvar como GIF
    # Nota: output_gif já vem como caminho completo do script de treino
    try:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0  # 0 = Loop infinito
        )
        print(f"GIF salvo com sucesso em: {os.path.abspath(output_gif)}")
    except Exception as e:
        print(f"Erro ao salvar GIF: {e}")
    
if __name__ == "__main__":
    # Teste manual (ajuste os caminhos se for rodar este arquivo sozinho)
    create_gif(
        input_folder="results/training_logs/dcgan_epochs", 
        output_gif="results/teste_manual.gif", 
        file_names="epoch_", 
        duration=100
    )