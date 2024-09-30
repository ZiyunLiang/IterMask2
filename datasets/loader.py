import random
import os
import torch as th
import numpy as np

from datasets.brats2021 import BraTS2021Dataset, get_brats2021_train_transform_abnormalty

def seed_worker(worker_id):
    np.random.seed(worker_id)
    random.seed(0)

g = th.Generator()
g.manual_seed(0)


def get_data_loader(dataset, data_path, config, mod, split_set='train', generator=True):
    if dataset == 'brats':
        loader = get_data_loader_brats(mod, data_path, config.model.training.batch_size, config.model.image_size,
                                           split_set=split_set)
    else:
        raise Exception("Dataset does exit")

    return get_generator_from_loader(loader) if generator else loader


def get_data_loader_brats(mod, path, batch_size, image_size, split_set: str = 'train'):

    assert split_set in ["train", "val", "test"]
    default_kwargs = {"drop_last": False, "batch_size": batch_size, "pin_memory": False, "num_workers": 1,
                      "prefetch_factor": 8, "worker_init_fn": seed_worker, "generator": g, }
    if split_set == "test":
        patient_dir = os.path.join(path, 'test')
        default_kwargs["shuffle"] = False
    elif split_set == "val":
        patient_dir = os.path.join(path, 'val')
        default_kwargs["shuffle"] = False
    else:
        patient_dir = os.path.join(path, 'train')
        default_kwargs["shuffle"] = True
        default_kwargs["num_workers"] = 8
    transforms = get_brats2021_train_transform_abnormalty(image_size)
    dataset = BraTS2021Dataset(
        data_root=patient_dir,
        mode='train',
        input_modality=mod,
        transforms=transforms)

    print(f"dataset lenght: {len(dataset)}")
    return th.utils.data.DataLoader(dataset, **default_kwargs)

def get_generator_from_loader(loader):
    while True:
        yield from loader
