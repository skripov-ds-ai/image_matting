from glob import glob
from tqdm import tqdm
from PIL import Image
mask_paths = glob("./data/Figaro1k/Figaro1k/GT/*/*.pbm")

for p in tqdm(mask_paths):
    img = Image.open(p)
    img.save(p.replace('.pbm', '.png'))

