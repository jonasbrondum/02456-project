import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image


class CansDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        #self.transforms = transforms
        # TODO Correct these paths
        self.imgs = list(sorted(os.listdir(os.path.join(root,"video1"))))
        self.bbox = list(sorted(os.listdir(os.path.join(root,"video1/frames"))))
    
    def __getitem__(self, idx):
        # load images and bboxes
        img_path = os.path.join(self.root, "frames", self.imgs[idx])
        bbox_path = os.path.join(self.root, "video1/frames", self.bbox[idx])
        img = Image.open(img_path).convert("RGB")
        num_objs = 2

        bbox = []
        label = []
        with open(bbox_path, 'r') as f:
            for line in f:
                line = line.split(" ")
                id = int(line[0]) # class label, 0=beer, 1=cola
                xmin = int(line[1])
                ymin = int(line[2])
                xmax = int(line[3])
                ymax = int(line[4])
                bbox.append([xmin, ymin, xmax, ymax])
                label.append(id)
        bbox = torch.as_tensor(bbox, dtype=torch.float32)
        labels = torch.as_tensor(label, dtype=torch.float32) #torch.ones((num_objs, ), dtype=torch.float32)
        image_id = torch.tensor([idx])
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])

        target = {}
        target["boxes"] = bbox
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area


        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    dataset = CansDataset('data/')