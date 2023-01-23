import numpy as np
from PIL import Image
from sklearn.neighbors import KDTree
import schem
import json
import pathlib
import timeit
import logging

print("Ładowanie ścieżki...")
path = str(pathlib.Path(__file__).parent.resolve()) + r"\\"
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def load_blocks(mode):
    logging.info("Ładowanie listy bloków")
    black_list = []
    blocks = {}
    blocks_raw = {}
    with open(path + "blocks.json", "r") as file:
        blocks_raw = json.load(file)
    for i in blocks_raw.keys():
        if i == "top" and 1 in mode:
            blocks.setdefault(i, {}).update(blocks_raw["top"])
        if i == "side" and 2 in mode:
            blocks.setdefault(i, {}).update(blocks_raw["side"])
        if i == "bottom" and 3 in mode:
            blocks.setdefault(i, {}).update(blocks_raw["bottom"])
        if i == "solid" and 4 in mode:
            blocks.setdefault(i, {}).update(blocks_raw["solid"])
        if 0 in mode:
            blocks.update(blocks_raw[i])
    for i in blocks.copy():
        for o in blocks[i].copy():
            if any(x in o for x in black_list):
                del blocks[i][o]
    return blocks


def get_closest_indices(blocks_lab, img_r):
    # Tworzenie drzewa KDTree z danych bloków
    tree = KDTree(blocks_lab)
    pixels = np.array(img_r)
    pixels = pixels.reshape(-1, 3)
    pixels_lab = np.array(
        [rgb_to_lab(pixel[0], pixel[1], pixel[2]) for pixel in pixels]
    )
    closest_indices = tree.query(pixels_lab, k=3)[1]

    return closest_indices


def get_invisible(img):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alfa = []
    width, height = img.size
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            if (pixels[x, y])[3] <= 100:
                alfa.append((x, y))
    return alfa


def image_prep(scale, img, Resampling, remove_bg):
    logging.info("Przetwarzanie obrazu...")

    img_d = Image.open(path + r"images\\" + img)
    width, height = img_d.size
    img_r = img_d.resize((round((width * scale) / height), scale), Resampling)
    alfa = []
    if img_r.mode != "RGB":
        if remove_bg:
            alfa = get_invisible(img_r)
        img_r = img_r.convert("RGB")
    width, height = img_r.size
    return img_r, width, height, alfa


def create_matrix(
    img: str, mode: list[int], scale: int, remove_bg: bool = False
):
    start = timeit.default_timer()
    blocks = load_blocks(mode)

    img_r, width, height, alfa = image_prep(
        scale, img, Image.Resampling.BICUBIC, remove_bg
    )

    logging.info("Tworzenie macierzy...")
    blocks_lab = np.array([blocks[block] for block in blocks.keys()])

    closest_indices = get_closest_indices(blocks_lab, img_r)

    closest_blocks = [list(blocks.keys())[i[0]] for i in closest_indices]

    coords = {}
    for item in list(set(closest_blocks)):
        for idx, element in enumerate(closest_blocks):
            if element == item:
                coords.setdefault(item, []).append(idx)

    closest_blocks = np.array(closest_blocks)

    # Wstawianie bloków na odpowiednie pozycje w nowym obrazie
    logging.info("Generowanie nowego obrazka...")
    result = Image.new("RGBA", (width * 16, height * 16))
    done = 1
    start_placing = timeit.default_timer()
    stop_placing = 0
    for element in coords.items():
        logging.info(
            "Done: {}/{}  Current: {} Left: {} seconds".format(done, width*height, element[0], round(((stop_placing) / (done)) * (width*height-done), 2)))
        img_block = Image.open(r"{}blocks\\{}".format(path, element[0]))
        for coord in element[1]:
            done += 1
            y = 0
            while coord >= width:
                coord -= width
                y += 1
            x = coord
            result.paste(img_block, (x * 16, y * 16))
        stop_placing = timeit.default_timer() - start_placing
    # for x in range(width):
    #     for y in range(height):
    #         index = y * width + x
    #         img_block = Image.open(rf"{path}blocks\\{closest_blocks[index]}")
    #         result.paste(img_block, (x * 16, y * 16))

    logging.info("Zapisywanie...")
    stop = timeit.default_timer() - start
    logging.info("{} sekund".format(stop))
    result.save("{}output\\{}png".format(path, img[:-3]))

    result.show()
    # schem.createScheamtic(matrix, img[5:-4])
    return


def rgb_to_lab(r, g, b):
    r = max(r, 0.1)
    g = max(g, 0.1)
    b = max(b, 0.1)
    r = r / 255
    g = g / 255
    b = b / 255
    r = r > 0.04045 and ((r + 0.055) / 1.055) ** 2.4 or r / 12.92
    g = g > 0.04045 and ((g + 0.055) / 1.055) ** 2.4 or g / 12.92
    b = b > 0.04045 and ((b + 0.055) / 1.055) ** 2.4 or b / 12.92
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883
    x = x > 0.008856 and x ** (1 / 3) or (7.787 * x) + 16 / 116
    y = y > 0.008856 and y ** (1 / 3) or (7.787 * y) + 16 / 116
    z = z > 0.008856 and z ** (1 / 3) or (7.787 * z) + 16 / 116
    L = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    return (L, a, b)


create_matrix("gura.jpg", [0], 1000, False)
