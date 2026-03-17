import os
import numpy as np
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from random import random, choice
from PIL import Image
from torchvision.transforms import InterpolationMode


class ImageSet(data.Dataset):

    def __init__(self, root_dir, set_type="train", aug=True, size=(256,256), mode='rcrop'):

        self.mode = mode
        self.size = size
        self.aug = aug

        if set_type == "train":
            inp_dir = os.path.join(root_dir, "in")
            gt_dir = os.path.join(root_dir, "gt")

        elif set_type == "val":
            inp_dir = os.path.join(root_dir, "val", "in")
            gt_dir = os.path.join(root_dir, "val", "gt")

        else:
            raise ValueError("set_type must be train or val")

        self.inp_list = []
        self.gt_list = []

        file_list = sorted(os.listdir(gt_dir))

        for f in file_list:

            if f.endswith("_gt.png"):

                gt_path = os.path.join(gt_dir, f)

                inp_name = f.replace("_gt.png", "_in.png")

                inp_path = os.path.join(inp_dir, inp_name)

                if os.path.exists(inp_path):

                    self.inp_list.append(inp_path)
                    self.gt_list.append(gt_path)

        self.num_samples = len(self.inp_list)

        assert self.num_samples > 0, "Dataset is empty!"

    def __len__(self):
        return self.num_samples


    def augs(self, inp, gt):

        if self.mode == 'rcrop':

            w, h = gt.size

            tl = np.random.randint(0, h - self.size[0])
            tt = np.random.randint(0, w - self.size[1])

            gt = torchvision.transforms.functional.crop(gt, tt, tl, self.size[0], self.size[1])
            inp = torchvision.transforms.functional.crop(inp, tt, tl, self.size[0], self.size[1])

        else:

            gt = torchvision.transforms.functional.resize(gt, self.size, InterpolationMode.BICUBIC)
            inp = torchvision.transforms.functional.resize(inp, self.size, InterpolationMode.BICUBIC)


        if random() < 0.5:

            inp = torchvision.transforms.functional.hflip(inp)
            gt = torchvision.transforms.functional.hflip(gt)

        if random() < 0.5:

            inp = torchvision.transforms.functional.vflip(inp)
            gt = torchvision.transforms.functional.vflip(gt)

        if random() < 0.5:

            angle = choice([90, 180, 270])

            inp = torchvision.transforms.functional.rotate(inp, angle)
            gt = torchvision.transforms.functional.rotate(gt, angle)

        return inp, gt


    def __getitem__(self, index):

        inp_data = Image.open(self.inp_list[index]).convert("RGB")
        gt_data = Image.open(self.gt_list[index]).convert("RGB")

        to_tensor = transforms.ToTensor()

        if self.aug:

            inp_data, gt_data = self.augs(inp_data, gt_data)

        else:

            if self.size is not None:

                inp_data = torchvision.transforms.functional.resize(inp_data, self.size, InterpolationMode.BICUBIC)
                gt_data = torchvision.transforms.functional.resize(gt_data, self.size, InterpolationMode.BICUBIC)


        return to_tensor(inp_data), to_tensor(gt_data)
