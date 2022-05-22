import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from utils import *
from model import YOLOv1
from datasets import VOCDataset
from loss import YoloLoss


LEARNING_RATE = 2E-5
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "check_point.pth"
IMG_DIR = 'data/images/'
ANN_DIR = 'data/labels/'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, box):
        for t in self.transforms:
            img = t(img)
        return img, box


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


def train(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        predictions = model(x)
        loss = loss_fn(predictions, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
    print("mean loss: {}".format(sum(mean_loss) / len(mean_loss)))


def main():
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    train_dataset = VOCDataset(
        "data/train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        ann_dir=ANN_DIR,
    )

    test_dataset = VOCDataset(
        "data/test.csv", transform=transform, img_dir=IMG_DIR, ann_dir=ANN_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    for epoch in range(EPOCHS):
        train(train_loader, model, optimizer, loss_fn)
        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")
        # pdb.set_trace()


if __name__ == "__main__":
    main()
