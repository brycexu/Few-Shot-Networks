import csv
import os
import shutil

# unzip -qo '/home/ubuntu/project/cifar100.zip' -d '/home/ubuntu/project/'

base_dir = '/home/ubuntu/project/'
source_dir =  base_dir + 'cifar100/data/'
dest_dir = base_dir + 'cifar100/images/'
split_dir = base_dir + 'cifar100/splits/bertinetto/'

def create_split_csv(split):
    """
    Converts the split file in the original CIFARFS r2d2 Bertinetto repo to
    .csv split files where first column is image name and second is image
    class name
    """
    split_file = open(split_dir + split + '.txt', 'r')
    classes = split_file.read().split('\n')
    with open(split_dir + split + '.csv', 'w', newline='') as file:
        # write first line
        writer = csv.writer(file)
        writer.writerow(["filename", "label"])
        for directory in os.listdir(source_dir):
            if directory in classes:
                for image in os.listdir(source_dir + directory):
                    # write image + class name
                    writer.writerow([image, directory])

create_split_csv('train')
create_split_csv('val')
create_split_csv('test')

if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

for direct in os.listdir(source_dir):
    for images in os.listdir(source_dir + direct):
        shutil.copyfile(source_dir + direct + "/" + images, dest_dir + images)
