import numpy as np
from PIL import Image
from sklearn.neighbors import KDTree
import schem
import json
import pathlib


path = str(pathlib.Path(__file__).parent.resolve())+r"\\"
blocks = {}
with open(path + "blocks.json", "r") as file:
    blocks = json.load(file)


def create_matrix(img: str, mode1: list[int], scale: int):

    blu = {}
    black_list = []
    print("Ładowanie listy bloków...")
    for i in blocks.keys():
        if i == "top" and 1 in mode1:
            blu.setdefault(i, {}).update(blocks["top"])
        if i == "side" and 2 in mode1:
            blu.setdefault(i, {}).update(blocks["side"])
        if i == "bottom" and 3 in mode1:
            blu.setdefault(i, {}).update(blocks["bottom"])
        if i == "solid" and 4 in mode1:
            blu.setdefault(i, {}).update(blocks["solid"])
        if 0 in mode1:
            blu.update(blocks[i])
        for i in blocks.copy():
            for o in blocks[i].copy():
                if any(x in o for x in black_list):
                    del blocks[i][o]

    img_d = Image.open(path + r"images\\" + img)
    width, height = img_d.size
    print("Ładowanie obrazu")
    img_r = img_d.resize(
        (round((width * scale) / height), scale), Image.Resampling.BICUBIC
    )
    if img_r.mode != "RGB":
        img_r = img_r.convert("RGB")
    width, height = img_r.size
    print("Generowanie nowego obrazka...")
    result = Image.new("RGB", (width * 16, height * 16))
    blocks_lab = np.array([blu[block] for block in blu.keys()])
    # Tworzenie drzewa KDTree z danych bloków
    tree = KDTree(blocks_lab)
    pixels = np.array(img_r)
    pixels = pixels.reshape(-1, 3)
    pixels_lab = np.array(
        [rgb_to_lab(pixel[0], pixel[1], pixel[2]) for pixel in pixels]
    )
    closest_indices = tree.query(pixels_lab, k=3)[1]

    closest_blocks = [list(blu.keys())[i[0]] for i in closest_indices]

    closest_blocks = np.array(closest_blocks)
    matrix = np.zeros((height, width))
    matrix = np.array(matrix, dtype=str)
    # Wstawianie bloków na odpowiednie pozycje w nowym obrazie
    for x in range(width):
        print(str(x+1) + "/" + str(width))
        for y in range(height):
            index = y * width + x
            for i in blocks:
                if closest_blocks[index] in blocks[i]:
                    face = i
                    img_block = Image.open(
                        fr"{path}blocks\\{face}\\{closest_blocks[index]}"
                    )
                    result.paste(img_block, (x * 16, y * 16))
                    matrix[y][x] = (
                        str(closest_blocks[index])[:-5]
                        if str(closest_blocks[index])[-5]
                        in ("1", "2", "3", "4", "5", "6", "7", "8", "9")
                        else str(closest_blocks[index])[:-4]
                    )
    print("Zapisywanie...")
    # with open(path + "//matrix.txt", "w") as mx:
    #     lm = matrix.tolist()
    #     for item in lm:
    #         mx.write("%s\n" % str(item))
    result.save(path + r"output\\" + img)
    # schem.createScheamtic(matrix, img[5:-4])
    return


def rgb_to_lab(r, g, b):
    if r <= 0.1:
        r = 1
    if g <= 0.1:
        g = 1
    if b <= 0.1:
        b = 1
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


create_matrix("image.png", [0], 78)
