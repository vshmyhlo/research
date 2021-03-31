from itertools import cycle

import numpy as np
from PIL import Image

colors = [
    {
        "name": "Light Pink",
        "hex": "ffadad",
        "rgb": [255, 173, 173],
        "cmyk": [0, 32, 32, 0],
        "hsb": [0, 32, 100],
        "hsl": [0, 100, 84],
        "lab": [79, 30, 12],
    },
    {
        "name": "Deep Champagne",
        "hex": "ffd6a5",
        "rgb": [255, 214, 165],
        "cmyk": [0, 16, 35, 0],
        "hsb": [33, 35, 100],
        "hsl": [33, 100, 82],
        "lab": [88, 8, 30],
    },
    {
        "name": "Lemon Yellow Crayola",
        "hex": "fdffb6",
        "rgb": [253, 255, 182],
        "cmyk": [1, 0, 29, 0],
        "hsb": [62, 29, 100],
        "hsl": [62, 100, 86],
        "lab": [98, -12, 35],
    },
    {
        "name": "Tea Green",
        "hex": "caffbf",
        "rgb": [202, 255, 191],
        "cmyk": [21, 0, 25, 0],
        "hsb": [110, 25, 100],
        "hsl": [110, 100, 87],
        "lab": [95, -28, 25],
    },
    {
        "name": "Celeste",
        "hex": "9bf6ff",
        "rgb": [155, 246, 255],
        "cmyk": [39, 4, 0, 0],
        "hsb": [185, 39, 100],
        "hsl": [185, 100, 80],
        "lab": [92, -25, -13],
    },
    {
        "name": "Baby Blue Eyes",
        "hex": "a0c4ff",
        "rgb": [160, 196, 255],
        "cmyk": [37, 23, 0, 0],
        "hsb": [217, 37, 100],
        "hsl": [217, 100, 81],
        "lab": [79, 2, -33],
    },
    {
        "name": "Maximum Blue Purple",
        "hex": "bdb2ff",
        "rgb": [189, 178, 255],
        "cmyk": [26, 30, 0, 0],
        "hsb": [249, 30, 100],
        "hsl": [249, 100, 85],
        "lab": [76, 20, -37],
    },
    {
        "name": "Mauve",
        "hex": "ffc6ff",
        "rgb": [255, 198, 255],
        "cmyk": [0, 22, 0, 0],
        "hsb": [300, 22, 100],
        "hsl": [300, 100, 89],
        "lab": [86, 30, -20],
    },
    {
        "name": "Baby Powder",
        "hex": "fffffc",
        "rgb": [255, 255, 252],
        "cmyk": [0, 0, 1, 0],
        "hsb": [60, 1, 100],
        "hsl": [60, 100, 99],
        "lab": [100, -1, 1],
    },
]


def fill(image):
    pass


image = np.zeros((900 * 2, 1600 * 2, 3), dtype=np.uint8)
ratio = 1 / 2

t, l, b, r = 0, 0, image.shape[0], image.shape[1]


def random_color():
    # return np.random.choice(colors)["rgb"]
    return np.random.randint(0, 255, size=(3,))


dirs = cycle(["r", "b", "l", "t"])

while (b - t) * (r - l) > 2:
    color = random_color()
    image[t:b, l:r] = color

    print(t, l, b, r)

    dir = next(dirs)

    if dir == "r":
        l = l + round((r - l) * ratio)
    elif dir == "b":
        t = t + round((b - t) * ratio)
    elif dir == "l":
        r = r - round((r - l) * ratio)
    elif dir == "t":
        b = b - round((b - t) * ratio)


Image.fromarray(image).save("./rec.png")
