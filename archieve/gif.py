import glob
from PIL import Image
import os

def create_gif(input_folder, output_gif, file_names, duration):
    # 1. Buscar todas as imagens que começam com o padrão definido
    # O sort() é vital aqui. Como usei {:03d} no nome do arquivo anterior (001, 002...),
    # a ordenação alfabética padrão funcionará corretamente.
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
        append_images=images[1:], # Adiciona o resto da lista
        duration=_duration,             # Duração de cada frame em milissegundos (100ms = 10fps)
        loop=0                    # 0 = Loop infinito
    )

    print(f"Sucesso! GIF salvo em: {os.path.abspath(output_file)}")

    # Opcional: Limpar as imagens estáticas após gerar o GIF
    # for f in filenames:
    #     os.remove(f)

def run_gif():
    # create_gif(input_folder="results_cats_dcgan", output_gif="dcgan_cat_epoch_.gif", file_names="dcgan_cat_epoch_", duration=20)
    # create_gif(input_folder="results_cats_dcgan_long", output_gif="dcgan_cat_epoch_.gif", file_names="dcgan_cat_epoch_", duration=82)
    create_gif(input_folder="results_cats_dcgan_long_afhq", output_gif="dcgan_cat_long__afhq_epoch_.gif", file_names="dcgan_cat_epoch_", duration=20)
    # create_gif(input_folder="results_cats_mlp", output_gif="cat_mlp_epoch_.gif", file_names="cat_mlp_epoch_", duration=25)
    # create_gif(input_folder="results_cats_wgan", output_gif="wgan_cat_epoch_.gif", file_names="wgan_cat_epoch_", duration=25)
    # create_gif(input_folder="results_cpu", output_gif="gan_generated_epoch_.gif", file_names="gan_generated_epoch_", duration=6)
    # create_gif(input_folder="results_dcgan_gpu", output_gif="dcgan_epoch_.gif", file_names="dcgan_epoch_", duration=2)
    # create_gif(input_folder="results_gpu", output_gif="gan_generated_epoch_.gif", file_names="gan_generated_epoch_", duration=6)
    
if __name__ == "__main__":
    create_gif(input_folder="results_cats_mlp", output_gif="cat_mlp_epoch_.gif", file_names="cat_mlp_epoch_", duration=25)
    # run_gif()