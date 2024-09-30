import os
import sys
import argparse
import random

import numpy as np
import torch as th
import blobfile as bf
from pathlib import Path

from datasets import loader
from configs import get_config
from utils import logger
from utils.script_util import create_model
from utils.metrics import sensitivity_metric, precision_metric, dice_score
sys.path.append(str(Path.cwd()))
from torchmetrics.functional import structural_similarity_index_measure
from models.test_model import iter_mask_refinement, validation_thres, iter_mask_refinement_bestthres

def normalize(img, _min=None, _max=None):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def main(args):
    use_gpus = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpus)
    config = get_config.file_from_dataset(args.dataset)

    if args.experiment_name_first_iter is None:
        experiment_name_first_iter = 'first_iter' + '_' + args.dataset + '_' + args.modality
    else:
        experiment_name_first_iter = args.experiment_name_first_iter
    if args.experiment_name_masked_autoencoder is None:
        experiment_name_masked_autoencoder = 'masked_autoencoder' + '_' + args.dataset + '_' + args.modality
    else:
        experiment_name_masked_autoencoder = args.experiment_name_masked_autoencoder
    experiment_name = experiment_name_masked_autoencoder
    logger.configure(Path(experiment_name),
                     format_strs=["log", "stdout", "csv", "tensorboard"])
    logger.log("creating loader...")
    input_mod = args.modality
    test_loader = loader.get_data_loader(args.dataset, args.data_dir, config, input_mod, split_set='test', generator=False)
    val_loader = loader.get_data_loader(args.dataset, args.data_dir, config, input_mod, split_set='val', generator=False)
    logger.log("creating model and diffusion...")

    model_first_iter = create_model(config, image_level_cond=False)
    model_masked_autoencoder = create_model(config, image_level_cond=True)


    filename = "model090000.pt"
    path_first_iter = bf.join('./model_save', experiment_name_first_iter, filename)
    path_masked_autoencoder = bf.join('./model_save', experiment_name_masked_autoencoder, filename)

    model_first_iter.load_state_dict(
        th.load(path_first_iter, map_location=th.device('cuda'))
    )
    model_first_iter.to(th.device('cuda'))
    model_masked_autoencoder.load_state_dict(
        th.load(path_masked_autoencoder, map_location=th.device('cuda'))
    )
    model_masked_autoencoder.to(th.device('cuda'))

    if config.model.use_fp16:
        model_first_iter.convert_to_fp16()

    model_first_iter.eval()
    model_masked_autoencoder.eval()

    logger.log("sampling...")

    num_sample = 0
    img_true_all = np.zeros((len(test_loader.dataset), config.model.num_input_channels*1, config.model.image_size,
             config.model.image_size))
    img_pred_all = np.zeros((len(test_loader.dataset), config.model.num_input_channels*1, config.model.image_size,
             config.model.image_size))
    img_pred_mask_all = np.zeros(
        (len(test_loader.dataset), config.model.num_input_channels * 1, config.model.image_size,
         config.model.image_size))
    brain_mask_all = np.zeros((len(test_loader.dataset), config.model.num_input_channels, config.model.image_size, config.model.image_size))
    test_data_seg_all = np.zeros((len(test_loader.dataset), config.model.num_input_channels,
                               config.model.image_size, config.model.image_size))
    max_dice_value_all = np.zeros((len(test_loader.dataset)))
    size_tumor_all = np.zeros((len(test_loader.dataset)))

    if args.best_threshold is False:
        val_iter_num = 0
        for val_data_dict in enumerate(val_loader):
            val_data_input = val_data_dict[1].pop('input').cuda()
            val_data_mask_inpaint = val_data_dict[1].pop('gauss_mask').cuda()
            val_data_brain_mask = val_data_dict[1].pop('brainmask').cuda()
            validation_thres(model_masked_autoencoder, val_data_input, val_data_mask_inpaint, val_data_brain_mask,
                             val_iter_num, experiment_name_masked_autoencoder)
            val_iter_num += 1

    num_iter = 0
    for test_data_dict in enumerate(test_loader):
        test_data_input = test_data_dict[1].pop('input').cuda()
        test_data_seg = test_data_dict[1].pop('seg')
        brain_mask = test_data_dict[1].pop('brainmask').cuda()
        test_data_seg = (th.ones(test_data_seg.shape) * (test_data_seg > 0)).cuda()
        if not args.best_threshold:
            final_mask, final_reconstruction = iter_mask_refinement(
                model_masked_autoencoder, model_first_iter, test_data_input,
                brain_mask, experiment_name_masked_autoencoder
            )
            img_true_all[num_sample:num_sample + test_data_input.shape[0]] = test_data_input.detach().cpu().numpy()
            img_pred_mask_all[num_sample:num_sample + test_data_input.shape[0]] = final_mask.cpu().numpy()
            img_pred_all[num_sample:num_sample + test_data_input.shape[0]] = final_reconstruction.cpu().numpy()
            brain_mask_all[num_sample:num_sample + test_data_input.shape[0]] = brain_mask.cpu().numpy()
            test_data_seg_all[num_sample:num_sample + test_data_input.shape[0]] = test_data_seg.cpu().numpy()
            size_tumor_all[num_sample:num_sample + test_data_input.shape[0]] = test_data_seg.sum(dim=(1, 2, 3)).cpu().numpy()
            num_sample += test_data_input.shape[0]
            num_iter += 1
        else:
            final_mask, final_reconstruction, dice_max = iter_mask_refinement_bestthres(
                model_masked_autoencoder, model_first_iter, test_data_input,
                brain_mask, test_data_seg
            )
            img_true_all[num_sample:num_sample + test_data_input.shape[0]] = test_data_input.detach().cpu().numpy()
            img_pred_mask_all[num_sample:num_sample + test_data_input.shape[0]] = final_mask.cpu().numpy()
            img_pred_all[num_sample:num_sample + test_data_input.shape[0]] = final_reconstruction.cpu().numpy()
            brain_mask_all[num_sample:num_sample + test_data_input.shape[0]] = brain_mask.cpu().numpy()
            test_data_seg_all[num_sample:num_sample + test_data_input.shape[0]] = test_data_seg.cpu().numpy()
            size_tumor_all[num_sample:num_sample + test_data_input.shape[0]] = test_data_seg.sum(dim=(1, 2, 3)).cpu().numpy()
            max_dice_value_all[num_sample:num_sample + test_data_input.shape[0]] = dice_max.cpu().numpy()
            num_sample += test_data_input.shape[0]

    logger.log("all the confidence maps from the testing set saved...")

    if args.best_threshold:
        max_dice_bestthres = np.mean(max_dice_value_all)
        logger.log(f"dice using the best threshold: {max_dice_bestthres}")

    if not args.best_threshold:
        error_map = normalize((img_true_all - img_pred_all) ** 2)*brain_mask_all
        dice = np.zeros(100)
        sensitivity = np.zeros(1)
        precision = np.zeros(1)
        logger.log("finding the best threshold...")
        for thres in range(100):
            mask_inpaint_input = np.where(thres / 10000 < error_map, 1.0, 0.0) * brain_mask_all
            for num in range(len(test_loader.dataset)):
                dice[thres] += dice_score(test_data_seg_all[num, 0, :, :], mask_inpaint_input[num, 0, :, :])

        dice = dice / len(test_loader.dataset)
        max_dice_index = np.argmax(dice)
        max_dice = dice[max_dice_index]

        dice_mask_large = 0
        dice_mask_medium = 0
        dice_mask_small = 0

        logger.log("computing the matrixs...")

        mask_inpaint_input = (np.where(max_dice_index / 10000 < error_map, 1.0, 0.0) * brain_mask_all)

        for num in range(len(test_loader.dataset)):
            pred_thre = mask_inpaint_input[num, 0, :, :]
            sensitivity += sensitivity_metric(pred_thre, test_data_seg_all[num, 0, :, :])
            precision += precision_metric(pred_thre, test_data_seg_all[num, 0, :, :])
            if size_tumor_all[num] < 200:
                dice_mask_small += dice_score(test_data_seg_all[num, 0, :, :], (mask_inpaint_input[num, 0, :, :]))
            elif size_tumor_all[num] >= 200 and size_tumor_all[num] < 800:
                dice_mask_medium += dice_score(test_data_seg_all[num, 0, :, :], (mask_inpaint_input[num, 0, :, :]))
            elif size_tumor_all[num] >= 800:
                dice_mask_large += dice_score(test_data_seg_all[num, 0, :, :], (mask_inpaint_input[num, 0, :, :]))


        sensitivity = sensitivity / len(test_loader.dataset)
        precision = precision / len(test_loader.dataset)
        img_pred_all = th.from_numpy(img_pred_all).cuda()
        brain_mask_all = th.from_numpy(brain_mask_all).cuda()
        test_data_seg_all = th.from_numpy(test_data_seg_all).cuda()
        img_true_all = th.from_numpy(img_true_all).cuda()
        len_large = len(size_tumor_all[size_tumor_all >= 800])
        len_midium = len(size_tumor_all[(size_tumor_all >= 200) & (size_tumor_all < 800)])
        len_small = len(size_tumor_all[size_tumor_all < 200])
        dice_mask_large = dice_mask_large / len_large
        dice_mask_medium = dice_mask_medium / len_midium
        dice_mask_small = dice_mask_small / len_small
        ssim = structural_similarity_index_measure(img_pred_all * brain_mask_all * (1-test_data_seg_all), img_true_all * brain_mask_all* (1-test_data_seg_all),
                                                         sigma=1.5, kernel_size=11,
                                                         data_range=(img_pred_all * brain_mask_all).max() - (
                                                                     img_pred_all * brain_mask_all).min(), k1=0.01, k2=0.03)


        logger.log(f"dice: {max_dice}, dice_small: {dice_mask_small},  dice_medium: {dice_mask_medium},  dice_large: {dice_mask_large}, sensitivity: {sensitivity}, precision: {precision}, "
                   f"ssim: {ssim}")



def reseed_random(seed):
    random.seed(seed)  # python random generator
    np.random.seed(seed)  # numpy random generator
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", help="the id of the gpu you want to use, like 0", type=int, default=1)
    parser.add_argument("--dataset", help="dataset name like brats", type=str, default='brats')
    parser.add_argument("--modality", help="input modality, choose from flair, t2, t1, t1ce", type=str, default='flair')
    parser.add_argument("--data_dir", help="data directory", type=str, default='./datasets/data')
    parser.add_argument("--experiment_name_first_iter", help="The file name to load for the first iteration model", type=str, default=None)
    parser.add_argument("--experiment_name_masked_autoencoder", help="The file name to load for the masked autoencoder model", type=str, default=None)
    parser.add_argument("--best_threshold", help="whether to compute the result using the best threshold or not", type=str2bool, default=False)
    args = parser.parse_args()
    print(args.dataset)
    main(args)
