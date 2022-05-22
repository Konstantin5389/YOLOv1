import torch
import pandas as pd
import os
from PIL import Image
import cv2
import numpy as np
import pdb


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, ann_dir, split_size=7, num_classes=20, num_boxes=2, transform=None) -> None:
        super(VOCDataset, self).__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_list = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list.iloc[index, 0])
        ann_path = os.path.join(self.ann_dir, self.img_list.iloc[index, 1])
        boxes = []
        with open(ann_path, 'r') as f:
            for line in f.readlines():
                boxes.append([float(x) if float(x) != int(float(x)) else int(float(x))
                              for x in line.strip().split()])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)
        if self.transform:
            image, boxes = self.transform(image, boxes)
        gt_mat = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, w, h = box.tolist()
            class_label = int(class_label)
            x_cell, y_cell = int(x * self.S), int(y * self.S)
            x_off, y_off = (x * self.S - x_cell), (y * self.S - y_cell)
            width_cell, height_cell = w * self.S, h * self.S
            box_coordinate = torch.tensor(
                [x_off, y_off, width_cell, height_cell])

            if gt_mat[y_cell, x_cell, 20] == 0:
                gt_mat[y_cell, x_cell, class_label] = 1
                gt_mat[y_cell, x_cell, 20] = 1
                gt_mat[y_cell, x_cell, 21:25] = box_coordinate

        return image, gt_mat


if __name__ == "__main__":
    csv_file = '/opt/datasets/ljy/PascalVOC_YOLO/train.csv'
    img_dir = '/opt/datasets/ljy/PascalVOC_YOLO/images/'
    ann_dir = '/opt/datasets/ljy/PascalVOC_YOLO/labels/'
    trainset = VOCDataset(csv_file, img_dir, ann_dir)
    image, ann = trainset[0]
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (448, 448))
    for i in range(trainset.S):
        for j in range(trainset.S):
            if ann[i, j, 20] == 0:
                continue
            x_off, y_off, w_cell, h_cell = ann[i, j, 21:25]
            x = (j + x_off) * 64
            y = (i + y_off) * 64
            w = w_cell * 64
            h = h_cell * 64
            image = cv2.rectangle(
                image,
                (int(x) - int(w) // 2, int(y) - int(h) // 2),
                (int(x) + int(w) // 2, int(y) + int(h) // 2),
                (0, 255, 0),
                4
            )
    cv2.imwrite('test.jpg', image)
