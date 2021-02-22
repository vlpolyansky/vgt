import numpy as np
import os
import array
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_folder', default='vanhateren/')
parser.add_argument('--out', default='vanhateren_images.npy')
args = parser.parse_args()

files = os.listdir(args.data_folder)
images = []
for filename in tqdm(files):
    with open(args.data_folder + '/' + filename, 'rb') as handle:
        s = handle.read()
        arr = array.array('H', s)
        arr.byteswap()
        img = np.array(arr, dtype='uint16').reshape(1024, 1536)
        images.append(img)
images = np.array(images)

np.save(args.out, images)
