import os
import random
import numpy as np
import monai.transforms as transforms
from os.path import join
from pathlib import Path
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset

def get_brats2021_train_transform_abnormalty(image_size):
    base_transform = [
        transforms.EnsureChannelFirstd(
            keys=['input', 'brainmask', 'seg', 'gauss_mask'], channel_dim='no_channel'),
        transforms.Resized(
            keys=['input', 'brainmask', 'seg', 'gauss_mask'],
            spatial_size=(image_size, image_size)),
    ]
    return transforms.Compose(base_transform)


class BraTS2021Dataset(Dataset):
    def __init__(self, data_root: str, mode: str, input_modality, transforms=None):
        super(BraTS2021Dataset, self).__init__()

        assert mode in ['train', 'test'], 'Unknown mode'
        self.mode = mode
        self.data_root = data_root
        self.input_modality = input_modality

        self.transforms = transforms
        self.case_names_input = sorted(list(Path(os.path.join(self.data_root, input_modality)).iterdir()))
        self.case_names_brainmask = sorted(list(Path(os.path.join(self.data_root, 'brainmask')).iterdir()))
        self.case_names_seg = sorted(list(Path(os.path.join(self.data_root, 'seg')).iterdir()))

    def __getitem__(self, index: int) -> tuple:
        name_input = self.case_names_input[index].name
        name_brainmask = self.case_names_brainmask[index].name
        name_seg = self.case_names_seg[index].name
        base_dir_input = join(self.data_root, self.input_modality, name_input)
        base_dir_brainmask = join(self.data_root, 'brainmask', name_brainmask)
        base_dir_seg = join(self.data_root, 'seg', name_seg)

        input = np.load(base_dir_input).astype(np.float32)
        brain_mask = np.load(base_dir_brainmask).astype(np.float32)
        seg = np.load(base_dir_seg).astype(np.float32)
        
        ### random mask generation ###
        num = random.randint(0, 2)
        covar11 = random.uniform(0.3, 10)
        covar22 = random.uniform(0.3, 10)
        covar12 = random.uniform(0, np.sqrt(covar11 * covar22)) * (-1 if random.randint(0, 1) == 0 else 1)
        target_area = np.where(brain_mask > 0)
        rand = np.random.randint(target_area[0].shape)
        if num == 0 or num == 1:
            x, y = np.mgrid[-12:12:.1, -12:12:.1]
            num1 = np.random.randint(np.sqrt(covar11 * covar22) * 100, np.sqrt(covar11 * covar22) * 200)
            num2 = np.random.randint(5, 10)
            num3 = np.random.randint(3, 10)
            mean1 = target_area[0][rand][0] / 240 * 24 - 12
            mean2 = target_area[1][rand][0] / 240 * 24 - 12
        elif num == 2:
            x, y = np.mgrid[-48:48:.4, -48:48:.4]
            num1 = np.random.randint(np.sqrt(covar11 * covar22) * 5, np.sqrt(covar11 * covar22) * 20)
            num2 = np.random.randint(5, 10)
            num3 = np.random.randint(3, 10)
            mean1 = target_area[0][rand][0] / 240 * 96 - 48
            mean2 = target_area[1][rand][0] / 240 * 96 - 48

        pos = np.dstack((x, y))

        rv = multivariate_normal([mean1, mean2], [[covar11, covar12], [covar12, covar22]])
        gau_pdf = rv.pdf(pos)
        gau_pdf = gau_pdf / gau_pdf.max()
        gau_pdf = gau_pdf

        p = gau_pdf[target_area[0], target_area[1]]
        l1 = 4
        l2 = 8
        l3 = 16
        l4 = 32

        unique_number_dim1 = np.random.choice(target_area[0].shape[0], num1 + num2 + num3, p=p.reshape(-1) / p.sum())
        unique_number_dim2 = np.random.choice(target_area[0].shape[0], num1 + num2 + num3, p=p.reshape(-1) / p.sum())
        gauss_mask = np.ones((input.shape[0], input.shape[1]))
        for i in range(unique_number_dim1.shape[0]):
            if i < num1:
                gauss_mask[target_area[0][unique_number_dim1[i]]:target_area[0][unique_number_dim1[i]]+l1, target_area[1][unique_number_dim2[i]]:target_area[1][unique_number_dim2[i]]+l1] = 0
            elif i < num1 + num2:
                gauss_mask[target_area[0][unique_number_dim1[i]]:target_area[0][unique_number_dim1[i]]+l2, target_area[1][unique_number_dim2[i]]:target_area[1][unique_number_dim2[i]]+l2] = 0
            elif i < num1 + num2 + num3:
                gauss_mask[target_area[0][unique_number_dim1[i]]:target_area[0][unique_number_dim1[i]]+l3, target_area[1][unique_number_dim2[i]]:target_area[1][unique_number_dim2[i]]+l3] = 0
            else:
                gauss_mask[target_area[0][unique_number_dim1[i]]:target_area[0][unique_number_dim1[i]]+l4, target_area[1][unique_number_dim2[i]]:target_area[1][unique_number_dim2[i]]+l4] = 0

        item = self.transforms(
            {'input': input, 'brainmask': brain_mask, 'seg': seg, 'gauss_mask': gauss_mask})

        return item

    def __len__(self):
        return len(self.case_names_input)

