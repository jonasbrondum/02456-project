import os
import torchvision
import numpy as np
from shutil import copyfile
from PIL import Image

def image_color_augmentation(root_dir):
    video_choice = 'video2/'
    bb_dir = root_dir + 'data/' + video_choice + 'boundingboxes/'
    frame_dir = root_dir + 'data/' + video_choice + 'frames/'
    data_files = [x[2] for x in os.walk(bb_dir)][0]

    for bb in data_files:
        img_name = bb[:-4] + '.jpg'

        img_orig = Image.open(frame_dir + img_name)

        img_aug = img_orig.copy()

        imgvals = np.random.rand(4)*0.5
        color_jitter = torchvision.transforms.ColorJitter(brightness=imgvals[0], contrast=imgvals[1], saturation=imgvals[2], hue=imgvals[3])
        img_aug = color_jitter(img_aug)
        img_aug = img_aug.save(frame_dir + bb[:-4] + 'aug.jpg')

        src_bb = bb_dir + bb
        dest_bb = bb_dir + bb[:-4] + 'aug.txt'
        copyfile(src_bb, dest_bb)

    print('----- Finished augmenting data -----')
root_dir = 'E:/Sync/Dokumenter/Universitet/Master/7_semester/02456_Deep_learning/02456-project/'
image_color_augmentation(root_dir)
