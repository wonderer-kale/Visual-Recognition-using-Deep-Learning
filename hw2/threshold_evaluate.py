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


def inference_with_thresholds(
        model, data_loader, thresholds, val_annotations,
        output_json_dir="predictions"):
    """Run inference for multiple thresholds and evaluate accuracy and mAP."""
    os.makedirs(output_json_dir, exist_ok=True)
    best_threshold = None
    best_accuracy = 0.0
    best_map = 0.0

    for threshold in thresholds:
        predictions = []

        with torch.no_grad():
            for images, ids in tqdm(
                    data_loader,
                    desc=f"Running inference (Threshold={threshold})"):
                images = [img.to(DEVICE) for img in images]
                outputs = model(images)

                for output, image_id in zip(outputs, ids):
                    for box, label, score in zip(
                            output["boxes"],
                            output["labels"],
                            output["scores"]):
                        if score < threshold:
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

        # Save predictions to a temporary JSON file
        temp_json_path = os.path.join(
            output_json_dir, f"pred_{threshold:.2f}.json")
        with open(temp_json_path, "w") as f:
            json.dump(predictions, f, indent=2)

        # Evaluate accuracy and mAP
        accuracy, map_score = evaluate_accuracy(
            temp_json_path, val_annotations)
        print(f"Threshold: {threshold:.2f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"mAP: {map_score:.4f}")

        # Update the best threshold
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_map = map_score
            best_threshold = threshold

    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best mAP: {best_map:.4f}")
    return best_threshold


def evaluate_accuracy(pred_json_path, val_annotations):
    """Evaluate accuracy and mAP using the validation annotations."""
    with open(pred_json_path, "r") as f:
        predictions = json.load(f)

    coco_gt = COCO(val_annotations)
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Return accuracy (or mAP@[IoU=0.50]) and mAP@[IoU=0.50:0.95]
    accuracy = coco_eval.stats[1]  # mAP@[IoU=0.50]
    map_score = coco_eval.stats[0]  # mAP@[IoU=0.50:0.95]
    return accuracy, map_score


def main():
    test_img_root = "nycu-hw2-data/valid"  # Use validation set
    val_annotations = "nycu-hw2-data/valid.json"
    checkpoint_path = "v2_checkpoints_half/model_epoch_9.pth"

    test_dataset = TestImageDataset(test_img_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    model = get_model(NUM_CLASSES, checkpoint_path=checkpoint_path)

    # Test thresholds from 0.5 to 0.9
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_threshold = inference_with_thresholds(
        model, test_loader, thresholds, val_annotations)

    print(f"Best threshold for accuracy: {best_threshold}")


if __name__ == "__main__":
    main()
