import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# load the model by .pth and print number of parameters
model = models.resnext50_32x4d(
    pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)
model.load_state_dict(torch.load('resnext50_32x4d_hw1.pth'))

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("#params: ", sum(p.numel()
                       for p in model.parameters() if p.requires_grad))

with open('prediction.csv', 'w') as f:
    f.write('image_name,pred_label\n')

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
for img_name in os.listdir('hw1-data/data/test'):
    img = Image.open(os.path.join('hw1-data/data/test', img_name))
    img = data_transforms(img).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, pred_label = torch.max(output, 1)
        # write img_name,pred_label to .csv file
        with open('prediction.csv', 'a') as f:
            f.write(f'{img_name},{pred_label.item()}\n')

with open('prediction.csv', 'r') as f:
    lines = f.readlines()
    lines = [line.replace('.jpg', '') for line in lines]
    with open('prediction.csv', 'w') as f:
        f.writelines(lines)
