import os
from multiprocessing import Pool

import click
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


@click.command()
@click.option("--input-path", type=click.Path(), required=True)
@click.option("--output-path", type=click.Path(), required=True)
@click.option("--num-workers", type=click.INT, default=os.cpu_count())
def main(input_path, output_path, num_workers):
    with Pool(num_workers) as pool:
        pool.starmap(
            preprocess_image,
            tqdm(
                [
                    (os.path.join(input_path, image_path), os.path.join(output_path, image_path))
                    for image_path in os.listdir(input_path)
                ]
            ),
        )


def preprocess_image(input_path, output_path):
    if os.path.exists(output_path):
        print("skipping {}".format(input_path))
        return

    try:
        image = Image.open(input_path)
        image.load()
    except UnidentifiedImageError:
        print("can't load {}".format(input_path))
        return
    except Exception as e:
        print("error {} for {}".format(e, input_path))
        return

    if image.mode != "RGB":
        image = image.convert("RGB")
        print("converting to RGB {}".format(input_path))

    if min(image.size) > 512:
        image = T.Resize(512)(image)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


if __name__ == "__main__":
    main()
