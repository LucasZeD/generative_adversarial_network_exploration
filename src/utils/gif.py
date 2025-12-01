import glob
from PIL import Image
import os

def create_gif(input_folder, output_gif, file_names, duration):
    # 1. Buscar todas as imagens que começam com o padrão especificado
    filenames = sorted(glob.glob(os.path.join(input_folder, file_names + "*.png")))
    _duration = duration

    if not filenames:
        print("Nenhuma imagem encontrada! Verifique se o treino rodou e salvou os .png.")
        return

    print(f"Encontradas {len(filenames)} imagens. Gerando GIF...")

    # 2. Carregar imagens na memória
    images = [Image.open(f) for f in filenames]

    # 3. Salvar como GIF
    output_file = os.path.join(input_folder, output_gif)
    
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=_duration,
        loop=0                    # 0 = Loop infinito
    )

    print(f"Sucesso! GIF salvo em: {os.path.abspath(output_file)}")
    
if __name__ == "__main__":
    create_gif(input_folder="results_cats_mlp", output_gif="cat_mlp_epoch_.gif", file_names="cat_mlp_epoch_", duration=25)