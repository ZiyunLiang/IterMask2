import os

import numpy as np
import torch as th
import torch.fft as fft
import blobfile as bf
from utils import logger
from utils.metrics import dice_score

def validation_thres(
        model,
        test_data_input,
        test_data_mask_inpaint,
        test_data_brain_mask,
        iter_num,
        experiment_name_masked_autoencoder
):
    with th.no_grad():
        y_input = fft.fftshift(fft.fft2(test_data_input))
        center = (test_data_input.shape[2] // 2, test_data_input.shape[3] // 2)
        X, Y = np.ogrid[:test_data_input.shape[2], :test_data_input.shape[3]]
        radius = 15
        dist_from_center1 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = th.from_numpy((dist_from_center1 < radius)).cuda()
        mask = ~mask
        y_masked = mask * y_input
        abs_masked = th.abs(y_masked)
        abs = th.abs(y_input)
        angle = th.angle(y_input)
        abs_ones = th.ones(abs.shape).cuda()
        abs_mask_zerotot1 = abs_masked * mask + abs_ones * ~mask
        fft_ = abs_mask_zerotot1 * th.exp((1j) * angle)
        img = fft.ifft2(fft.ifftshift(fft_))
        x_mask_real = th.real(img)
        x_cond = x_mask_real

        noise = th.randn_like(test_data_input)
        mask_data = (1 - test_data_mask_inpaint) * test_data_brain_mask
        input = (1 - mask_data) * test_data_input + mask_data * noise
        input = th.cat((input, x_cond), 1)
        x0_pred_forward = input
        x0_pred_backward = model(x0_pred_forward.float())

        loss = ((test_data_input - x0_pred_backward) ** 2)
        if iter_num != 0:
            loss_load = th.load(os.path.join(logger.get_dir(), experiment_name_masked_autoencoder+'_loss_save.pt'))
            loss_all = th.cat((loss_load, loss), dim=0)
            th.save(loss_all, os.path.join(logger.get_dir(), experiment_name_masked_autoencoder+'_loss_save.pt'))
        else:
            th.save(loss, os.path.join(logger.get_dir(), experiment_name_masked_autoencoder+'_loss_save.pt'))
        if iter_num != 0:
            mask_load = th.load(os.path.join(logger.get_dir(), experiment_name_masked_autoencoder+'_mask_save.pt'))
            mask_all = th.cat((mask_load, mask_data), dim=0)
            th.save(mask_all, os.path.join(logger.get_dir(), experiment_name_masked_autoencoder+'_mask_save.pt'))
        else:
            th.save(mask_data, os.path.join(logger.get_dir(), experiment_name_masked_autoencoder+'_mask_save.pt'))

def iter_mask_refinement(
        model_masked_autoencoder,
        model_firststep,
        test_data_input,
        brain_mask,
        experiment_name_masked_autoencoder
):
    with th.no_grad():
        y_input = fft.fftshift(fft.fft2(test_data_input))
        center = (test_data_input.shape[2] // 2, test_data_input.shape[3] // 2)
        X, Y = np.ogrid[:test_data_input.shape[2], :test_data_input.shape[3]]
        radius = 15
        dist_from_center1 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = th.from_numpy((dist_from_center1 < radius)).cuda()
        mask = ~mask
        y_masked = mask * y_input

        abs_masked = th.abs(y_masked)
        abs = th.abs(y_input)
        angle = th.angle(y_input)
        abs_ones = th.ones(abs.shape).cuda()
        abs_mask_zerotot1 = abs_masked * mask + abs_ones * ~mask
        fft_ = abs_mask_zerotot1 * th.exp((1j) * angle)
        img = fft.ifft2(fft.ifftshift(fft_))
        x_mask_real = th.real(img)
        x_cond = x_mask_real

        ### loading error maps from validation set (for threshold choosing) ###
        mask_all_val = th.load(os.path.join(logger.get_dir(), experiment_name_masked_autoencoder+'_mask_save.pt'))
        loss_all_val = th.load(os.path.join(logger.get_dir(), experiment_name_masked_autoencoder+'_loss_save.pt'))

        loss_masked = loss_all_val * mask_all_val
        kthnum = mask_all_val.shape[0] * mask_all_val.shape[2] * mask_all_val.shape[3] - mask_all_val.sum() * 0.20
        thres_validation = th.kthvalue(loss_masked.flatten(), kthnum.int()).values

        flag = th.ones([test_data_input.shape[0]]).cuda()
        final_reconstruction = th.zeros_like(test_data_input)
        final_mask = th.zeros_like(test_data_input)
        mask_inpaint_input = brain_mask
        i = 0
        while flag.sum() != 0:
            if i == 0:
                x0_pred_firststep = model_firststep(x_cond)
                error_map = ((test_data_input - x0_pred_firststep) ** 2) * brain_mask
                thres = th.zeros([test_data_input.shape[0]]).cuda()
                for num in range(test_data_input.shape[0]):
                    kthnum = brain_mask.shape[2] * brain_mask.shape[3] - brain_mask[num, :, :, :].sum() * 0.6
                    thres[num] = th.kthvalue(error_map[num, 0, :, :].flatten(), kthnum.int()).values
                thres = thres.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, test_data_input.shape[2], test_data_input.shape[3])
            else:
                thres = thres_validation
            mask_inpaint_input_new = th.where(thres < error_map, 1.0, 0.0) * brain_mask

            ratio = (mask_inpaint_input.sum(dim=(1, 2, 3)) - mask_inpaint_input_new.sum(
                dim=(1, 2, 3))) / mask_inpaint_input.sum(dim=(1, 2, 3))
            ratio = th.where(th.isnan(ratio), -1, ratio)

            update_flag = (ratio < 0.01)*(flag == 1)
            flag = flag * (~update_flag).int()
            if i > 0:
                ### save the final predictions ###
                index = th.where((update_flag == 1).int())
                final_reconstruction[index] = x0_pred[index]
                final_mask[index] = mask_inpaint_input[index]
            if flag.sum() == 0:
                break

            mask_inpaint_input = mask_inpaint_input_new

            noise = th.randn_like(test_data_input)
            x_masked = (1 - mask_inpaint_input) * test_data_input + mask_inpaint_input * noise
            x_input = th.cat((x_masked, x_cond), dim=1)

            x0_pred = model_masked_autoencoder(x_input.float())

            x0_pred_combine = mask_inpaint_input * x0_pred + (1 - mask_inpaint_input) * test_data_input
            error_map = ((test_data_input - x0_pred_combine) ** 2)
            i += 1

        return final_mask, final_reconstruction

def iter_mask_refinement_bestthres(
        model_masked_autoencoder,
        model_firststep,
        test_data_input,
        brain_mask,
        test_data_seg=None
):
            with th.no_grad():
                y_input = fft.fftshift(fft.fft2(test_data_input))
                center = (test_data_input.shape[2] // 2, test_data_input.shape[3] // 2)
                X, Y = np.ogrid[:test_data_input.shape[2], :test_data_input.shape[3]]
                radius = 15
                dist_from_center1 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
                mask = th.from_numpy((dist_from_center1 < radius)).cuda()
                mask = ~mask
                y_masked = mask * y_input
                abs_masked = th.abs(y_masked)
                abs = th.abs(y_input)
                angle = th.angle(y_input)
                abs_ones = th.ones(abs.shape).cuda()
                abs_mask_zerotot1 = abs_masked * mask + abs_ones * ~mask
                fft_ = abs_mask_zerotot1 * th.exp((1j) * angle)
                img = fft.ifft2(fft.ifftshift(fft_))
                x_mask_real = th.real(img)
                x_cond = x_mask_real

                x0_pred_firststep = model_firststep(x_cond)
                error_map_first = ((test_data_input - x0_pred_firststep) ** 2) * brain_mask

                dice_all_rat = th.zeros((test_data_input.shape[0], 6))
                mask_plot_all_final = th.zeros((test_data_input.shape[0], test_data_input.shape[1],
                                                test_data_input.shape[2], test_data_input.shape[3])).cuda()
                img_pred_all_final = th.zeros((test_data_input.shape[0], test_data_input.shape[1],
                                                test_data_input.shape[2], test_data_input.shape[3])).cuda()
                mask_plot_all_rat = th.zeros((6, test_data_input.shape[0], test_data_input.shape[1],
                                              test_data_input.shape[2], test_data_input.shape[3])).cuda()
                img_plot_all_rat = th.zeros((6, test_data_input.shape[0], test_data_input.shape[1],
                                            test_data_input.shape[2], test_data_input.shape[3])).cuda()
                for rat in range(6):
                    error_map = error_map_first
                    thres = th.zeros([test_data_input.shape[0]]).cuda()
                    for i in range(test_data_input.shape[0]):
                        kthnum = brain_mask.shape[2] * brain_mask.shape[3] - brain_mask[i, :, :, :].sum() * (
                                    0.9 - rat / 10)
                        thres[i] = th.kthvalue(error_map[i, 0, :, :].flatten(), kthnum.int()).values

                    thres_firststep = thres.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1,
                                                                                      test_data_input.shape[2],
                                                                                      test_data_input.shape[3])

                    dice_all = th.zeros((test_data_input.shape[0], 10000)).cuda()
                    flag = th.ones([test_data_input.shape[0]]).cuda()
                    mask_inpaint_input = brain_mask

                    i = 0
                    while (thres < 0.3).sum() > 0:
                        if i == 0:
                            thres = thres_firststep
                        else:
                            thres = thres + update_flag.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1,
                                                                                      test_data_input.shape[2],
                                                                                      test_data_input.shape[3]) * 0.001
                        mask_inpaint_input_new = th.where(thres < error_map, 1.0, 0.0) * brain_mask

                        ratio = (mask_inpaint_input.sum(dim=(1, 2, 3)) - mask_inpaint_input_new.sum(
                            dim=(1, 2, 3))) / mask_inpaint_input.sum(dim=(1, 2, 3))
                        ratio = th.where(th.isnan(ratio), -1, ratio)

                        update_flag = (ratio < 0.01)
                        flag = flag * (~update_flag).int()

                        mask_inpaint_input = mask_inpaint_input_new

                        noise = th.randn_like(test_data_input)
                        x_masked = (1 - mask_inpaint_input) * test_data_input + mask_inpaint_input * noise

                        x_input = th.cat((x_masked, x_cond), dim=1)

                        x0_pred = model_masked_autoencoder(x_input.float())

                        x0_pred_combine = mask_inpaint_input * x0_pred + (
                                    1 - mask_inpaint_input) * test_data_input

                        error_map = ((test_data_input - x0_pred_combine) ** 2)

                        for num in range(test_data_seg.shape[0]):
                            dice_all[num, i] = dice_score(test_data_seg[num, 0, :, :], mask_inpaint_input[num, 0, :, :])

                        if i == 0:
                            mask_inpaint_input_all = mask_inpaint_input.unsqueeze(0)
                            x0_pred_combine_all = x0_pred.unsqueeze(0)
                        else:
                            mask_inpaint_input_all = th.cat((mask_inpaint_input_all, mask_inpaint_input.unsqueeze(0)), dim=0)
                            x0_pred_combine_all = th.cat((x0_pred_combine_all, x0_pred.unsqueeze(0)), dim=0)
                        i += 1


                    dice_all_rat[:, rat] = th.max(dice_all, dim=1)[0]
                    dice_all_rat_max_index = th.argmax(dice_all, dim=1)
                    for k in range(test_data_input.shape[0]):
                        mask_plot_all_rat[rat, k, :, :, :] = mask_inpaint_input_all[dice_all_rat_max_index[k], k, :, :, :]
                        img_plot_all_rat[rat, k, :, :, :] = x0_pred_combine_all[dice_all_rat_max_index[k], k, :, :, :]

                dice_max = th.max(dice_all_rat, dim=1)[0]
                dice_all_whole_max_index = th.argmax(dice_all_rat, dim=1)

                for k in range(test_data_input.shape[0]):
                    mask_plot_all_final[k, :, :, :] = mask_plot_all_rat[dice_all_whole_max_index[k], k, :, :, :]
                    img_pred_all_final[k, :, :, :] = img_plot_all_rat[dice_all_whole_max_index[k], k, :, :, :]

                return mask_plot_all_final, img_pred_all_final, dice_max

