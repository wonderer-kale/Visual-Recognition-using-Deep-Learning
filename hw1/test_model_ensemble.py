import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

state_dict_list = ['resnext50_32x4d_hw1_0.pth',
                   'resnext50_32x4d_hw1_1.pth',
                   'resnext50_32x4d_hw1_2.pth',
                   'resnext50_32x4d_hw1_3.pth',
                   'resnext50_32x4d_hw1_4.pth',
                   'resnext101_32x8d_hw1_0.pth',
                   'resnext101_32x8d_hw1_1.pth',
                   'resnext101_32x8d_hw1_2.pth',
                   'resnext101_32x8d_hw1_3.pth',
                   'resnext101_32x8d_hw1_4.pth',]
trained_models = []
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
for state_dict in state_dict_list:
    if state_dict.startswith('resnext50'):
        model = models.resnext50_32x4d(pretrained=False)
    else:
        model = models.resnext101_32x8d(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    model.load_state_dict(torch.load(state_dict))
    model = model.to(device)
    model.eval()
    trained_models.append(model)

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with open('prediction.csv', 'w') as f:
    f.write('image_name,pred_label\n')

for img_name in os.listdir('hw1-data/data/test'):
    img = Image.open(os.path.join('hw1-data/data/test', img_name))
    img = data_transforms(img).unsqueeze(0).to(device)
    pred_count = [0.0] * 100
    with torch.no_grad():
        for i, model in enumerate(trained_models):
            output = model(img).to(device)
            _, pred_label = torch.max(output, 1)
            pred_count[pred_label.item()] += 1
        pred_label = torch.tensor(pred_count).argmax()

        # write img_name,pred_label to .csv file
        with open('prediction.csv', 'a') as f:
            f.write(f'{img_name},{pred_label.item()}\n')

with open('prediction.csv', 'r') as f:
    lines = f.readlines()
    lines = [line.replace('.jpg', '') for line in lines]
    with open('prediction.csv', 'w') as f:
        f.writelines(lines)
