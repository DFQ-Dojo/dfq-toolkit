"""
Create randomized labels for COCO images
COCO labels are structured as:
[0-79] x y w h
where x,y,w,h are normalized to 0-1 and have 6 places after decimal and 1 place before decimal (0/1)
e.g:
1 0.128828 0.375258 0.249063 0.733333
0 0.476187 0.289613 0.028781 0.138099

To randomize:
First generate width and height dimensions
Then jitter the x/y labels
Then fix using max/min clipping
"""

import numpy as np
import argparse
import os
from PIL import Image
from tqdm import tqdm
import random
from utils.general import init_seeds

MINDIM = 0.2
MAXDIM = 0.8


def populate(args):

    # folder
    os.makedirs(os.path.join(args.outdir, "images", "train2017"))
    os.makedirs(os.path.join(args.outdir, "labels", "train2017"))
    train_imgs = []
    for imgIdx in tqdm(range(args.numImages)):
        target_num = random.randint(1, 25)

        im = Image.new(mode="RGB", size=(256, 256), color=(127, 127, 127))

        # save
        outfile = "COCO_train2017_{:012d}".format(imgIdx + 1)
        im.save(os.path.join(args.outdir, "images", "train2017", outfile + ".jpg"))

        for i in range(target_num):
            # box: w,h,x,y
            width = MINDIM + (MAXDIM - MINDIM) * np.random.rand()
            height = MINDIM + (MAXDIM - MINDIM) * np.random.rand()
            x = 0.5 + (0.5 - width / 2.0) * np.random.rand() * np.random.choice([1, -1])
            y = 0.5 + (0.5 - height / 2.0) * np.random.rand() * np.random.choice([1, -1])
            assert x + width / 2.0 <= 1.0, "overflow width, x+width/2.0={}".format(x + width / 2.0)
            assert y + height / 2.0 <= 1.0, "overflow height, y+height/2.0={}".format(y + height / 2.0)

            # class
            cls = np.random.choice(np.arange(args.numClasses))
            _label_str = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(int(cls), x, y, width, height)
            with open(os.path.join(args.outdir, "labels", "train2017", outfile + ".txt"), "at") as f:
                f.write(_label_str)
        train_imgs.append(f"./images/train2017/{outfile}.jpg")
    with open(os.path.join(args.outdir, "train2017.txt"), "w") as fout:
        for img in train_imgs:
            line = img + "\n"
            fout.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="populate single box per image labels")
    parser.add_argument("--numImages", type=int, default=120, help="number of images to generate")
    parser.add_argument("--numClasses", type=int, default=80, help="number of classes")
    parser.add_argument("--outdir", type=str, required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")

    args = parser.parse_args()

    init_seeds(args.seed + 1, deterministic=True)
    populate(args)
