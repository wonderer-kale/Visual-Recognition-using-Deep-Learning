import os
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation


class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {
            'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2,
            'derain': 3, 'dehaze': 4, 'deblur': 5,
            'rain': 6, 'snow': 7
        }

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type:
            self._init_clean_ids()
        if 'denoise_25' in self.de_type:
            self._init_clean_ids()
        if 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'rain' in self.de_type or 'snow' in self.de_type:
            self._init_rain_snow_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        ref_file = self.args.data_file_dir + "noisy/denoise_airnet.txt"
        temp_ids = []
        temp_ids += [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for
                      id_ in name_list if id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x, "de_type": 0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x, "de_type": 1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x, "de_type": 2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_rain_snow_ids(self):
        self.rain_ids = []
        self.snow_ids = []

        degraded_dir = os.path.join(
            self.args.data_file_dir, "train", "degraded")
        clean_dir = os.path.join(self.args.data_file_dir, "train", "clean")

        if 'rain' in self.de_type:
            rain_files = [
                f for f in os.listdir(degraded_dir) if f.startswith("rain-")]
            for f in rain_files:
                degraded_path = os.path.join(degraded_dir, f)
                clean_name = f.replace("rain-", "rain_clean-")
                clean_path = os.path.join(clean_dir, clean_name)
                self.rain_ids.append({
                    "degraded_id": degraded_path,
                    "clean_id": clean_path,
                    "de_type": self.de_dict["rain"]})
            print(f"Loaded {len(self.rain_ids)} rain samples")

        if 'snow' in self.de_type:
            snow_files = [
                f for f in os.listdir(degraded_dir) if f.startswith("snow-")]
            for f in snow_files:
                degraded_path = os.path.join(degraded_dir, f)
                clean_name = f.replace("snow-", "snow_clean-")
                clean_path = os.path.join(clean_dir, clean_name)
                self.snow_ids.append({
                    "degraded_id": degraded_path,
                    "clean_id": clean_path,
                    "de_type": self.de_dict["snow"]})
            print(f"Loaded {len(self.snow_ids)} snow samples")

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.data_file_dir + "hazy/hazy_outside.txt"
        temp_ids += [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]

        self.hazy_counter = 0

        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"
        temp_ids += [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 120

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size,
                        ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size,
                        ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split(
            "rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids
        if "dehaze" in self.de_type:
            self.sample_ids += self.hazy_ids
        if 'rain' in self.de_type:
            self.sample_ids += self.rain_ids
        if 'snow' in self.de_type:
            self.sample_ids += self.snow_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(
                np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch = np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(
                    Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(
                    Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(
                    Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(
                    Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == self.de_dict["rain"]:
                # Load directly from provided paths
                degrad_img = crop_img(np.array(
                    Image.open(sample["degraded_id"]).convert('RGB')), base=16)
                clean_img = crop_img(np.array(
                    Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = os.path.basename(sample["clean_id"]).split('.')[0]
            elif de_id == self.de_dict["snow"]:
                # Load directly from provided paths
                degrad_img = crop_img(np.array(
                    Image.open(sample["degraded_id"]).convert('RGB')), base=16)
                clean_img = crop_img(np.array(
                    Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = os.path.basename(sample["clean_id"]).split('.')[0]

            degrad_patch, clean_patch = random_augmentation(
                *self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)
