import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from pycocotools import mask as coco_mask

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Define paths
BASE_DIR = "hw3-data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TEST_IDS_PATH = os.path.join(BASE_DIR, "test_image_name_to_ids.json")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


# Define data transforms
def get_transform():
    transforms = []
    transforms.append(T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


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
        model.rpn.anchor_generator.sizes = (
            (16,), (32,), (64,), (128,), (256,))
        # model.rpn.anchor_generator.aspect_ratios = ((0.5, 1.0, 2.0),) * 5
        model.rpn.anchor_generator.aspect_ratios = (
            (0.25, 0.5, 1.0, 2.0, 4.0),) * 5

    return model


num_classes = len(CLASS_MAP) + 1  # +1 for background


# Test inference function
def test_inference(half_anchor=False):
    # Load best model
    model = get_model(num_classes, half_anchor)
    model.load_state_dict(torch.load("best_model.pth"))
    model.to(DEVICE)
    model.eval()

    # Process test images
    results = []
    test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.tif')]

    for file_name in tqdm(test_files, desc="Processing test images"):
        image_id = test_image_name_to_ids.get(file_name)

        # Load and process image
        image_path = os.path.join(TEST_DIR, file_name)
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Transform image
        image_tensor = torch.as_tensor(
            image_np, dtype=torch.float32).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        # Get predictions
        with torch.no_grad():
            prediction = model(image_tensor)[0]

        # Process results
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        masks = prediction["masks"].cpu().numpy()

        # Apply score threshold
        score_threshold = 0.05
        keep = scores > score_threshold

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        masks = masks[keep]

        # Convert to submission format
        for box, score, label, mask in zip(boxes, scores, labels, masks):
            mask = mask[0] > 0.5
            mask = np.asfortranarray(mask.astype(np.uint8))

            rle = coco_mask.encode(mask)
            rle['counts'] = rle['counts'].decode('utf-8')

            x1, y1, x2, y2 = box.tolist()

            results.append({
                "image_id": image_id,
                "bbox": [x1, y1, x2, y2],
                "score": score.item(),
                "category_id": label.item(),
                "segmentation": {
                    "size": list(mask.shape),
                    "counts": rle['counts']
                }
            })

    # Save results
    with open("test-results.json", 'w') as f:
        json.dump(results, f)

    print(f"Results saved to results.json with {len(results)} predictions")


test_inference(half_anchor=False)
