import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20) -> None:
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S, self.B, self.C = split_size, num_boxes, num_classes
        self.noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):
        predictions = predictions.reshape(
            (-1, self.S, self.S, self.C + self.B * 5))
        iou1 = intersection_over_union(
            predictions[..., 21:25], targets[..., 21:25])
        iou2 = intersection_over_union(
            predictions[..., 26:30], targets[..., 21:25])
        ious = torch.cat((iou1, iou2), dim=-1)
        iou_maxes, best_boxes = torch.max(ious, dim=-1)
        best_boxes = best_boxes.unsqueeze(-1)
        exist_boxes = targets[..., 20:21]

        # box loss
        box_predictions = exist_boxes * (
            best_boxes * predictions[..., 26:30] +
            (1 - best_boxes) * predictions[..., 21:25]
        )

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_target = exist_boxes * targets[..., 21:25]
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_target, end_dim=-2)
        )

        # obj loss
        obj_predictions = (
            best_boxes * predictions[..., 25:26] +
            (1 - best_boxes) * predictions[..., 20:21]
        )
        obj_target = targets[..., 20:21]
        obj_loss = self.mse(
            torch.flatten(exist_boxes * obj_predictions, end_dim=-2),
            torch.flatten(exist_boxes * obj_target, end_dim=-2)
        )

        # no obj loss
        no_obj_loss = self.mse(
            torch.flatten((1 - exist_boxes) *
                          predictions[..., 20:21], end_dim=-2),
            torch.flatten((1 - exist_boxes) * targets[..., 20:21], end_dim=-2)
        )
        no_obj_loss += self.mse(
            torch.flatten((1 - exist_boxes) *
                          predictions[..., 25:26], end_dim=-2),
            torch.flatten((1 - exist_boxes) * targets[..., 20:21], end_dim=-2)
        )

        # class loss
        class_loss = self.mse(
            torch.flatten(exist_boxes * predictions[..., 0:20]),
            torch.flatten(exist_boxes * targets[..., 0:20])
        )

        loss = (
            self.lambda_coord * box_loss +
            obj_loss +
            self.noobj * no_obj_loss +
            class_loss
        )

        return loss


if __name__ == "__main__":
    criterion = YoloLoss()
    prediction = torch.randn((2, 7, 7, 30))
    targets = torch.zeros(30)
    targets[0] = 1
    targets[20] = 1
    targets[21:25] = torch.tensor([100, 100, 100, 100])
    targets = targets.repeat((2, 7, 7, 1))
    prediction = prediction.reshape((-1, 1470))
    print(criterion(prediction, targets))
