import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11  # 10 digits + background


class DigitDataset(Dataset):
    def __init__(self, root, json_file, transforms=None):
        self.root = root
        self.coco = COCO(json_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(image_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels = [], []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        image = F.to_tensor(image)

        return image, target

    def __len__(self):
        return len(self.ids)


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        pretrained=True)
    anchor_generator = AnchorGenerator(
        # default: ((32, 64, 128, 256, 512),)
        sizes=((16,), (32,), (64,), (128,), (256,)),
        # default: ((0.5, 1.0, 2.0),)
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model.rpn.anchor_generator = anchor_generator
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, epoch):
    model.train()
    total_loss = 0
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch}"):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
    return total_loss / len(data_loader)


def evaluate(model, data_loader, coco_gt, epoch):
    model.eval()
    coco_preds = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                image_id = int(target["image_id"])
                for box, label, score in zip(
                        output["boxes"], output["labels"], output["scores"]):
                    x1, y1, x2, y2 = box.tolist()
                    coco_preds.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score)
                    })

    with open("pred.json", "w") as f:
        json.dump(coco_preds, f)

    coco_dt = coco_gt.loadRes("pred.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP = coco_eval.stats[0]
    return mAP


def task2_predict():
    with open("pred.json", "r") as f:
        preds = json.load(f)

    image_dict = {}
    for pred in preds:
        img_id = pred["image_id"]
        if pred["score"] < 0.5:
            continue
        if img_id not in image_dict:
            image_dict[img_id] = []
        image_dict[img_id].append(
            (pred["bbox"][0], pred["category_id"]))  # sort by x_min

    rows = []
    for image_id in sorted(set(p["image_id"] for p in preds)):
        if image_id in image_dict:
            digits = sorted(image_dict[image_id], key=lambda x: x[0])
            label = ''.join(str(d[1]-1) for d in digits)
        else:
            label = "-1"
        rows.append({"image_id": image_id, "pred_label": label})

    pd.DataFrame(rows).to_csv("pred.csv", index=False)


def main():
    train_ds = DigitDataset("nycu-hw2-data/train", "nycu-hw2-data/train.json")
    val_ds = DigitDataset("nycu-hw2-data/valid", "nycu-hw2-data/valid.json")

    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    model = get_model(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    coco_gt = COCO("nycu-hw2-data/valid.json")
    record = []

    for epoch in range(1, 11):
        loss = train_one_epoch(model, optimizer, train_loader, epoch)
        mAP = evaluate(model, val_loader, coco_gt, epoch)

        task2_predict()
        df = pd.read_csv("pred.csv")
        gt = {ann["image_id"]: "" for ann in coco_gt.anns.values()}
        for ann in coco_gt.anns.values():
            gt[ann["image_id"]] += str(ann["category_id"] - 1)
        acc = np.mean(
            [str(gt.get(int(row.image_id), "-1")) ==
             str(row.pred_label) for _, row in df.iterrows()])

        record.append(
            {"epoch": epoch, "loss": loss, "mAP": mAP, "accuracy": acc})
        pd.DataFrame(record).to_csv("v2_record_half.csv", index=False)

        # save the model checkpoint
        torch.save(
            model.state_dict(), f"v2_checkpoints_half/model_epoch_{epoch}.pth")

    print("Training finished.")


if __name__ == "__main__":
    main()
