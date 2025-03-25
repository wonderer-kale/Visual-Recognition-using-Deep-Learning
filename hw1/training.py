import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Data preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hw1-data/data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                             shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
idx_to_class = {v: k for k, v in image_datasets['train'].class_to_idx.items()}

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Build model
model_ft = models.resnext50_32x4d(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)


# Train model
def train_model(model, criterion, optimizer, scheduler=None, num_epochs=25):
    train_acc_his = []
    train_loss_his = []
    val_acc_his = []
    val_loss_his = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase],
                                       desc=f'{phase} phase'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # convert the labels to the corresponding class
                labels = torch.tensor([int(idx_to_class[label.item()])
                                       for label in labels]).to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if phase == 'train' and scheduler is not None:
                        scheduler.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > 0.86:
                torch.save(model.state_dict(), 'resnext50_32x4d_hw1_'
                           + "%.4f" % epoch_acc + '.pth')

            if phase == 'train':
                train_acc_his.append(epoch_acc.item())
                train_loss_his.append(epoch_loss)
            else:
                val_acc_his.append(epoch_acc.item())
                val_loss_his.append(epoch_loss)

        print()

    return model, train_acc_his, val_acc_his, train_loss_his, val_loss_his


model_ft, acc_his, val_acc_his, loss_his, val_loss_his = train_model(
    model_ft, criterion, optimizer_ft, num_epochs=30
)
# Plot the learning accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(acc_his, label='Train Accuracy')
plt.plot(val_acc_his, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Accuracy Curve')
plt.legend()
plt.savefig('resnext50_32x4d_hw1_acc_curve.png')

# Plot the learning loss curve
plt.figure(figsize=(10, 5))
plt.plot(loss_his, label='Train Loss')
plt.plot(val_loss_his, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Loss Curve')
plt.legend()
plt.savefig('resnext50_32x4d_hw1_loss_curve.png')
