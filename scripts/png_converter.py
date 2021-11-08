from PIL import Image
import os

directory = 'C:/Sync/Dokumenter/Universitet/Master/7_semester/02456_Deep_learning/project20/data/video1/frames/'

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        im = Image.open(filename)
        name = filename[:-4]+'.png'
        rgb_im = im.convert('RGB')
        rgb_im.save(name)
        print(os.path.join(directory, filename))
        continue
    else:
        continue