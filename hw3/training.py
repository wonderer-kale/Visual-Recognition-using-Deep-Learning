import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from torchvision.ops import masks_to_boxes
from torch.utils.data import Dataset, DataLoader, random_split
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import skimage.io as sio

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Define paths
BASE_DIR = "hw3-data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TEST_IDS_PATH = os.path.join(BASE_DIR, "test_image_name_to_ids.json")
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Define class mapping
CLASS_MAP = {
    'class1': 1,
    'class2': 2,
    'class3': 3,
    'class4': 4
}

# Load test image name to ids mapping
with open(TEST_IDS_PATH, 'r') as f:
    test_image_data = json.load(f)
    test_image_name_to_ids = {entry["file_name"]: entry["id"] for
                              entry in test_image_data}


# Define dataset class
class CellSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.folders = [f for f in os.listdir(root_dir) if
                        os.path.isdir(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder_name = self.folders[idx]
        folder_path = os.path.join(self.root_dir, folder_name)

        # Load image
        image_path = os.path.join(folder_path, "image.tif")
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        masks = []
        labels = []

        for class_name, class_id in CLASS_MAP.items():
            class_path = os.path.join(folder_path, f"{class_name}.tif")
            if os.path.exists(class_path):
                mask = np.array(sio.imread(class_path))

                # Connected component analysis to separate instances
                num_labels, labeled_mask = cv2.connectedComponents(
                    mask.astype(np.uint8))

                for label_id in range(1, num_labels):  # Skip background (0)
                    binary_mask = (labeled_mask == label_id).astype(np.uint8)
                    masks.append(binary_mask)
                    labels.append(class_id)

        # Convert everything to tensors
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Get bounding boxes from masks
        boxes = masks_to_boxes(masks)

        # Filter out masks and boxes with zero area
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        valid_indices = areas > 0
        boxes = boxes[valid_indices]
        masks = masks[valid_indices]
        labels = labels[valid_indices]

        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([idx])
        target["area"] = areas[valid_indices]
        target["iscrowd"] = torch.zeros((len(masks),), dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, target


# Define data transforms
def get_transform():
    transforms = []
    transforms.append(T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


# Create dataset
full_dataset = CellSegmentationDataset(TRAIN_DIR, transform=None)

# Split into train and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Training on {train_size} samples, validating on {val_size} samples")

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    collate_fn=lambda x: tuple(zip(*x))
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    collate_fn=lambda x: tuple(zip(*x))
)


# Define model
def get_model(num_classes, half_anchor=False):
    # Load pretrained model
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=True)

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    if half_anchor:
        # default sizes are ((32,), (64,), (128,), (256,), (512,))
        model.rpn.anchor_generator.sizes = (
            (16,), (32,), (64,), (128,), (256,))
        model.rpn.anchor_generator.aspect_ratios = (
            (0.25, 0.5, 1.0, 2.0, 4.0),) * 5

    return model


# Initialize model
num_classes = len(CLASS_MAP) + 1  # +1 for background
model = get_model(num_classes, half_anchor=True)
model.to(DEVICE)

# Define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)


# COCO evaluation function
def evaluate_coco(model, data_loader, device):
    model.eval()
    coco_gt = create_coco_format(data_loader.dataset)
    coco_dt = []

    for images, targets in tqdm(data_loader, desc="Validation"):
        images = list(img.to(device) for img in images)

        with torch.no_grad():
            outputs = model(images)

        for i, output in enumerate(outputs):
            image_id = targets[i]["image_id"].item()
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks = output["masks"].cpu().numpy()

            # filter out scores less than 0.05
            keep = scores > 0.05
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            masks = masks[keep]

            for box, score, label, mask in zip(boxes, scores, labels, masks):
                mask = mask[0] > 0.5
                rle = coco_mask.encode(np.array(
                    mask, dtype=np.uint8, order='F'))
                rle['counts'] = rle['counts'].decode('utf-8')

                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1

                coco_dt.append({
                    'image_id': image_id,
                    'category_id': label.item(),
                    'bbox': [x1.item(), y1.item(), w.item(), h.item()],
                    'score': score.item(),
                    'segmentation': rle
                })

    coco_gt_obj = COCO()
    coco_gt_obj.dataset = coco_gt
    coco_gt_obj.createIndex()

    coco_dt_obj = coco_gt_obj.loadRes(coco_dt)
    coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # Return mAP@IoU=0.5:0.95


# Helper function to create COCO format for evaluation
def create_coco_format(dataset):
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add categories
    for class_name, class_id in CLASS_MAP.items():
        coco_dict["categories"].append({
            "id": class_id,
            "name": class_name
        })

    ann_id = 0

    # Add images and annotations
    for idx in range(len(dataset)):
        image, target = dataset[idx]

        image_id = target["image_id"].item()
        coco_dict["images"].append({
            "id": image_id,
            "width": image.shape[2],
            "height": image.shape[1]
        })

        boxes = target["boxes"].cpu().numpy()
        labels = target["labels"].cpu().numpy()
        masks = target["masks"].cpu().numpy()

        for box, label, mask in zip(boxes, labels, masks):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            rle = coco_mask.encode(np.array(mask, dtype=np.uint8, order='F'))
            rle['counts'] = rle['counts'].decode('utf-8')

            area = float((mask > 0).sum())

            coco_dict["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": area,
                "segmentation": rle,
                "iscrowd": 0
            })

            ann_id += 1

    return coco_dict


# Training function
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()

    running_loss = 0.0
    for images, targets in tqdm(data_loader, desc="Training"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    return running_loss / len(data_loader)


# Training loop
num_epochs = 10
best_map = 0.0

# store loss and map record for plotting
losses = []
maps = []

print(f"Starting training for {num_epochs} epochs")

for epoch in range(num_epochs):
    # Train for one epoch
    loss = train_one_epoch(model, optimizer, train_loader, DEVICE)
    losses.append(loss)

    # Evaluate on validation set
    map_score = evaluate_coco(model, val_loader, DEVICE)
    maps.append(map_score)

    print(f"Epoch: {epoch+1}/{num_epochs}")
    print(f"Loss: {loss:.4f}, mAP: {map_score:.4f}")

    # Save best model
    if map_score > best_map:
        best_map = map_score
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Saved best model with mAP: {best_map:.4f}")

# save loss and map records
losses_df = pd.DataFrame(losses, columns=["Loss"])
maps_df = pd.DataFrame(maps, columns=["mAP"])
losses_df.to_csv("losses.csv", index=False)
maps_df.to_csv("maps.csv", index=False)
print("Training completed!")
