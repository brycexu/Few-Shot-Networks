# Author : Bryce Xu
# Time : 2019/12/17
# Function: 

from __future__ import print_function
import csv
import glob
import os

from PIL import Image

path_to_images = '/mnt/datadev_2/std/mini-imagenet/images/'

all_images = glob.glob(path_to_images + '*')

# Resize images
for i, image_file in enumerate(all_images):
    im = Image.open(image_file)
    im = im.resize((84, 84), resample=Image.LANCZOS)
    im.save(image_file)
    if i % 500 == 0:
        print(i)

# Put in correct directory
for datatype in ['train', 'val', 'test']:
    os.system('mkdir ' + datatype)

    with open(datatype + '.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        last_label = ''
        for i, row in enumerate(reader):
            if i == 0:  # skip the headers
                continue
            label = row[1]
            image_name = row[0]
            if label != last_label:
                cur_dir = datatype + '/' + label + '/'
                os.system('mkdir ' + cur_dir)
                last_label = label
            os.system('mv images/' + image_name + ' ' + cur_dir)