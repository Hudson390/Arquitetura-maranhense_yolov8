import json
import os


def convert_vgg_to_yolo(vgg_json_path, output_dir, class_mapping, image_dimensions):
    """
    Converte anotações do formato VGG JSON para YOLO.

    Args:
        vgg_json_path (str): Caminho para o arquivo JSON do VGG.
        output_dir (str): Diretório onde os arquivos YOLO serão salvos.
        class_mapping (dict): Dicionário de mapeamento de classes {label: class_id}.
        image_dimensions (dict): Dicionário com as dimensões das imagens {filename: (width, height)}.

    """
    # Carregar o JSON do VGG
    with open(vgg_json_path, "r", encoding="utf-8") as f:
        vgg_data = json.load(f)

    # Criar o diretório de saída, se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Processar cada imagem no JSON
    for filename, annotations in vgg_data.items():
        if "regions" not in annotations or not annotations["regions"]:
            continue

        # Obter dimensões da imagem
        width, height = image_dimensions.get(filename, (None, None))
        if width is None or height is None:
            print(f"Dimensões da imagem {filename} não foram fornecidas. Pulando.")
            continue

        # Criar o arquivo .txt para a imagem
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_filepath = os.path.join(output_dir, txt_filename)

        with open(txt_filepath, "w") as txt_file:
            for region in annotations["regions"].values():  # Use .values() para iterar corretamente sobre os objetos
                # Obter as coordenadas do polígono
                shape_attributes = region["shape_attributes"]
                if shape_attributes["name"] != "polygon":
                    print(f"Formato inesperado para {filename}. Pulando.")
                    continue

                all_points_x = shape_attributes["all_points_x"]
                all_points_y = shape_attributes["all_points_y"]

                # Calcular o bounding box
                x = min(all_points_x)
                y = min(all_points_y)
                w = max(all_points_x) - x
                h = max(all_points_y) - y

                # Calcular as coordenadas centralizadas e normalizadas
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                norm_width = w / width
                norm_height = h / height

                # Obter o class_id
                label = region["region_attributes"].get("label")
                class_id = class_mapping.get(label, None)
                if class_id is None:
                    print(f"Classe '{label}' não encontrada no mapeamento. Pulando.")
                    continue

                # Escrever a linha no formato YOLO
                txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

    print(f"Conversão concluída! Arquivos salvos em: {output_dir}")


# Exemplo de uso
if __name__ == "__main__":
    vgg_json_path = "converter/labels_my-project-name_2025-01-18-12-59-22.json"
    output_dir = "converter"
    class_mapping = {
        "calçada": 0,
        "porta": 1,
        "grade": 2,
    }
    image_dimensions = {
        "2013-08-11_09-40-46_HDR.jpg": (1200, 1600),
    }

    convert_vgg_to_yolo(vgg_json_path, output_dir, class_mapping, image_dimensions)
