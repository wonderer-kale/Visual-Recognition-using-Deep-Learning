import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
import torchvision.models as models
import torchvision.transforms as transforms

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def calculate_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(float('inf'))
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.l1_loss = nn.L1Loss()
        self.vgg = self._get_vgg()
        self.vgg_loss_weight = 0.5
        self.ssim_loss_weight = 0.2
        self.l1_loss_weight = 1.0

        # Preprocessing for VGG
        self.vgg_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _get_vgg(self):
        vgg = models.vgg16(pretrained=True).features[:16]  # Up to conv3_3
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg.eval()

    def _vgg_loss(self, x, y):
        x = torch.clamp(x, 0, 1)
        y = torch.clamp(y, 0, 1)
        x = self.vgg_transform(x)
        y = self.vgg_transform(y)
        return self.l1_loss(self.vgg(x), self.vgg(y))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        # Clamp to [0, 1] for perceptual + SSIM
        restored = torch.clamp(restored, 0, 1)
        clean_patch = torch.clamp(clean_patch, 0, 1)

        # Compute losses
        l1 = self.l1_loss(restored, clean_patch)
        vgg = self._vgg_loss(restored, clean_patch)
        ssim_loss = 1 - ssim(restored, clean_patch,
                             data_range=1.0, size_average=True)

        total_loss = (
            self.l1_loss_weight * l1 +
            self.vgg_loss_weight * vgg +
            self.ssim_loss_weight * ssim_loss
        )

        self.log_dict({
            "train_l1": l1,
            "train_vgg": vgg,
            "train_ssim": ssim_loss,
            "train_total_loss": total_loss,
        })

        return total_loss

    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        restored = torch.clamp(restored, 0, 1)
        clean_patch = torch.clamp(clean_patch, 0, 1)

        # Losses
        l1 = self.l1_loss(restored, clean_patch)
        vgg = self._vgg_loss(restored, clean_patch)
        ssim_loss = 1 - ssim(restored,
                             clean_patch, data_range=1.0, size_average=True)
        psnr = calculate_psnr(restored, clean_patch)

        total_loss = (
            self.l1_loss_weight * l1 +
            self.vgg_loss_weight * vgg +
            self.ssim_loss_weight * ssim_loss
        )

        self.log_dict({
            "val_psnr": psnr,
            "val_l1": l1,
            "val_vgg": vgg,
            "val_ssim": ssim_loss,
            "val_total_loss": total_loss,
        }, prog_bar=True)

        return total_loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=4e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=350)
        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    # trainset = PromptTrainDataset(opt)
    full_dataset = PromptTrainDataset(opt)
    val_split = 0.1  # 10% for validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    trainset, valset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])
    # Save every 10 epochs (regardless of metric)
    periodic_ckpt = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        filename="epoch={epoch:03d}",
        every_n_epochs=10,
        save_top_k=-1  # Save ALL every 10 epochs
    )
    # Save best model based on highest PSNR
    best_ckpt = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        filename="best-epoch={epoch:03d}-val_psnr={val_psnr:.2f}",
        monitor="val_psnr",
        mode="max",
        save_top_k=1,
        save_last=True
    )
    trainloader = DataLoader(
        trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
        drop_last=True, num_workers=opt.num_workers)
    valloader = DataLoader(
        valset, batch_size=opt.batch_size, pin_memory=True, shuffle=False,
        drop_last=False, num_workers=opt.num_workers)

    model = PromptIRModel()

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger, callbacks=[periodic_ckpt, best_ckpt])
    trainer.fit(
        model=model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == '__main__':
    main()
