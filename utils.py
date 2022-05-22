from collections import Counter
import torch
import pdb


def intersection_over_union(pred_box, label_box, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = pred_box[..., 0:1] - pred_box[..., 2:3] / 2
        box1_y1 = pred_box[..., 1:2] - pred_box[..., 3:4] / 2
        box1_x2 = pred_box[..., 0:1] + pred_box[..., 2:3] / 2
        box1_y2 = pred_box[..., 1:2] + pred_box[..., 3:4] / 2

        box2_x1 = label_box[..., 0:1] - label_box[..., 2:3] / 2
        box2_y1 = label_box[..., 1:2] - label_box[..., 3:4] / 2
        box2_x2 = label_box[..., 0:1] + label_box[..., 2:3] / 2
        box2_y2 = label_box[..., 1:2] + label_box[..., 3:4] / 2
    elif box_format == "cornerpoint":
        box1_x1 = pred_box[..., 0:1]
        box1_y1 = pred_box[..., 1:2]
        box1_x2 = pred_box[..., 2:3]
        box1_y2 = pred_box[..., 3:4]
        box2_x1 = label_box[..., 0:1]
        box2_y1 = label_box[..., 1:2]
        box2_x2 = label_box[..., 2:3]
        box2_y2 = label_box[..., 3:4]
    else:
        raise ValueError("box_format should be 'midpoint' or 'cornerpoint'")

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    return inter / (area1 + area2 - inter + 1e-6)


def non_max_suppression(boxes, iou_threshold=0.5, conf_threshold=0.1, box_format="cornerpoint"):
    """NMS

    Args:
        boxes (_type_): [[classes, conf, x1, y1, x2, y2]]
    """
    boxes_after_nms = []
    boxes = [box for box in boxes if box[1] >= conf_threshold]
    boxes.sort(key=lambda x: x[1], reverse=True)
    while boxes:
        chosen_box = boxes.pop(0)
        boxes = [
            box for box in boxes
            if box[0] != chosen_box[0] or
            intersection_over_union(
                torch.tensor(box[2:]),
                torch.tensor(chosen_box[2:]),
                box_format=box_format
            ) < iou_threshold
        ]
        boxes_after_nms.append(chosen_box)
    return boxes_after_nms


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20, box_format="midpoint"):
    """_summary_

    Args:
        pred_boxes (_type_): [[img_id, class, conf, x1, y1, x2, y2]]
        true_boxes (_type_): same with pred_boxes
        iou_threshold (float, optional): Defaults to 0.5.
        num_classes (int, optional): Defaults to 20.
    """

    average_precision = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = [pred_box for pred_box in pred_boxes if pred_box[1] == c]
        ground_truths = [
            true_box for true_box in true_boxes if true_box[1] == c]
        gt_boxes_per_img = Counter(gt[0] for gt in ground_truths)
        for key, val in gt_boxes_per_img.items():
            gt_boxes_per_img[key] = torch.zeros(val)
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros_like(TP)
        num_all_ground_truth = len(ground_truths)
        if num_all_ground_truth == 0:
            continue
        for idx, detection in enumerate(detections):
            gt_img = [ground_truth for ground_truth in ground_truths
                      if ground_truth[0] == detection[0]]
            best_iou = 0
            best_iou_idx = 0
            for gt_idx, gt in enumerate(gt_img):
                if intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format) > best_iou:
                    best_iou = intersection_over_union(
                        torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)
                    best_iou_idx = gt_idx
            if best_iou >= iou_threshold:
                if gt_boxes_per_img[detection[0]][best_iou_idx] == 0:
                    TP[idx] = 1
                    gt_boxes_per_img[detection[0]][best_iou_idx] = 1
                else:
                    FP[idx] = 1
            else:
                FP[idx] = 1
        TP_cum = torch.cumsum(TP, dim=0)
        FP_cum = torch.cumsum(FP, dim=0)
        recalls = TP_cum / (num_all_ground_truth + epsilon)
        precisions = TP_cum / (TP_cum + FP_cum + epsilon)

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precision.append(torch.trapz(precisions, recalls))
    return sum(average_precision) / len(average_precision)


def convert_cellboxes(predictions, S=7):
    predictions = predictions.to('cpu')
    predictions = predictions.reshape(predictions.shape[0], S, S, -1)
    box1 = predictions[..., 21:25]
    box2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20:21], predictions[..., 25:26]), dim=-1)
    boxes = (1 - torch.argmax(scores, dim=-1).unsqueeze(-1)) * \
        box1 + torch.argmax(scores, dim=-1).unsqueeze(-1) * box2
    x_cell = torch.arange(S)
    y_cell = torch.arange(S)
    y_cell, x_cell = torch.meshgrid(y_cell, x_cell)
    cell = torch.cat((x_cell.unsqueeze(-1), y_cell.unsqueeze(-1)), dim=-1)
    cell = cell.repeat((predictions.shape[0], 1, 1, 1))
    boxes[..., 0:2] = boxes[..., 0:2] + cell
    boxes = boxes / S
    score, _ = torch.max(scores, dim=-1, keepdim=True)
    class_label = torch.argmax(predictions[..., :20], dim=-1, keepdim=True)
    return torch.cat((class_label, score, boxes), dim=-1)


def cellbox_box(out, S=7):
    out = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    out[..., 0:1] = out[..., 0:1].long()
    all_boxes = []
    for batch_idx in range(out.shape[0]):
        boxes = []
        for box_idx in range(S * S):
            boxes.append([x.item() for x in out[batch_idx, box_idx, :]])
        all_boxes.append(boxes)
    return all_boxes


def get_bboxes(loader, model, iou_threshold, threshold, pred_format="cells", box_format="midpoint", device="cuda"):
    """_summary_

    Args:
        loader (_type_): _description_
        model (_type_): _description_
        iou_threshold (_type_): _description_
        threshold (_type_): _description_
        pred_format (str, optional): _description_. Defaults to "cells".
        box_format (str, optional): _description_. Defaults to "midpoint".
        device (str, optional): _description_. Defaults to "cuda".
    """
    all_pred_boxes = []
    all_true_boxes = []
    model.eval()
    train_idx = 0
    for batch_idx, (x, target) in enumerate(loader):
        x = x.to(device)
        target.to(device)
        predictions = model(x)
        predictions = cellbox_box(predictions)
        labels = cellbox_box(target)
        for idx in range(len(predictions)):
            prediction = predictions[idx]
            label = labels[idx]
            nms_boxes = non_max_suppression(
                prediction,
                iou_threshold=iou_threshold,
                conf_threshold=threshold,
                box_format="midpoint"
            )
            for box in nms_boxes:
                all_pred_boxes.append([train_idx] + box)
            for box in label:
                all_true_boxes.append([train_idx] + box)
            train_idx += 1
    return all_pred_boxes, all_true_boxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


if __name__ == "__main__":
    boxes = [
        [1, 0.6, 50, 50, 150, 150],
        [1, 0.9, 40, 40, 160, 160],
        [1, 0.3, 0, 0, 200, 200],
        [0, 0.1, 40, 40, 120, 120],
        [0, 0.05, 0, 0, 200, 200],
    ]
    print(non_max_suppression(boxes))

    predictions = [
        [0, 0, 0.9, 50, 50, 150, 150],
        [0, 0, 0.6, 50, 50, 150, 150],
        [1, 0, 0.6, 40, 40, 160, 160]
    ]

    ground_truth = [
        [0, 0, 0.9, 50, 50, 150, 150],
        [1, 0, 0.6, 50, 50, 150, 150]
    ]
    print(mean_average_precision(predictions, ground_truth))

    cell_prediction = torch.randn((2, 7, 7, 30))
    print(convert_cellboxes(cell_prediction).shape)
