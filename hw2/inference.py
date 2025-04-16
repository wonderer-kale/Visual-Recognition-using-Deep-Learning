import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11  # 10 digits + background


class TestImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(".png")
        ])

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        image_id = int(os.path.splitext(file_name)[0])
        image_path = os.path.join(self.image_dir, file_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image)

        return image_tensor, image_id

    def __len__(self):
        return len(self.image_files)


def get_model(num_classes, checkpoint_path=None):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        pretrained=False)
    anchor_generator = AnchorGenerator(
        # default: ((32, 64, 128, 256, 512),)
        sizes=((16,), (32,), (64,), (128,), (256,)),
        # default: ((0.5, 1.0, 2.0),)
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model.rpn.anchor_generator = anchor_generator
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model.to(DEVICE)
    model.eval()
    return model


def inference(model, data_loader, output_json_path="pred.json"):
    predictions = []

    with torch.no_grad():
        for images, ids in tqdm(data_loader, desc="Running inference"):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for output, image_id in zip(outputs, ids):
                for box, label, score in zip(
                        output["boxes"], output["labels"], output["scores"]):
                    if score < 0.5:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        "image_id": int(image_id),
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score)
                    })

    # Sort predictions by image_id
    predictions.sort(key=lambda x: x["image_id"])

    with open(output_json_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved Task 1 predictions to {output_json_path}")
    return predictions


def generate_task2(predictions, output_csv_path="pred.csv"):
    image_dict = {}
    for pred in predictions:
        img_id = pred["image_id"]
        if pred["score"] < 0.5:
            continue
        if img_id not in image_dict:
            image_dict[img_id] = []
        image_dict[img_id].append(
            (pred["bbox"][0], pred["category_id"]))  # sort by x_min

    rows = []
    for image_id in range(1, 13069):
        if image_id in image_dict:
            digits = sorted(image_dict[image_id], key=lambda x: x[0])
            label = ''.join(str(d[1]-1) for d in digits)
        else:
            label = "-1"
        rows.append({"image_id": image_id, "pred_label": label})

    pd.DataFrame(rows).to_csv(output_csv_path, index=False)
    print(f"Saved Task 2 predictions to {output_csv_path}")


def main():
    test_img_root = "nycu-hw2-data/test"
    checkpoint_path = "v2_checkpoints_half/model_epoch_9.pth"

    test_dataset = TestImageDataset(test_img_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    model = get_model(NUM_CLASSES, checkpoint_path=checkpoint_path)
    predictions = inference(
        model, test_loader, output_json_path="v2_checkpoints_half/pred.json")
    generate_task2(predictions, output_csv_path="v2_checkpoints_half/pred.csv")


if __name__ == "__main__":
    main()
