import copy
import os
import time

import blobfile as bf
import torch as th
import torch.fft as fft
import numpy as np
from models.nn import mean_flat
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from utils import logger
from utils.fp16_util import MixedPrecisionTrainer
from models.nn import update_ema


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            data,
            batch_size,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            weight_decay=0.0,
            lr_decay_steps=0,
            lr_decay_factor=1,
            iterations: int = 80e4,
            num_input_channels=None,
            image_size=None,
            device=None,
            args=None
    ):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.weight_decay = weight_decay
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_factor = lr_decay_factor
        self.iterations = iterations
        self.num_input_channels = num_input_channels
        self.image_size = image_size

        log_dir = os.path.join('../logs_loss/', 'train')
        self.writer = SummaryWriter(log_dir=log_dir)
        '''timing'''
        self.args = args
        self.step = 0
        self.time_iter_start = 0
        self.forward_backward_time = 0
        self.device = device
        self.x0_pred = None
        self.recursive_flag = 0
        self.resume_step = 0
        self.sync_cuda = th.cuda.is_available()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        self.ema_params = [
            copy.deepcopy(self.mp_trainer.master_params)
            for _ in range(len(self.ema_rate))
        ]

    def run_loop(self):
        while (
                not self.lr_decay_steps
                or self.step + self.resume_step < self.iterations
        ):
            data_dict = next(self.data)
            self.run_step(data_dict)
            if self.step % self.save_interval == 0:
                self.save()
            if self.step % self.log_interval == 0:
                self.time_iter_end = time.time()
                if self.time_iter_start == 0:
                    self.time_iter = 0
                else:
                    self.time_iter = self.time_iter_end - self.time_iter_start
                self.log_step()
                logger.dumpkvs()
                self.time_iter_start = time.time()
            self.step += 1
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, data_dict):
        self.forward_backward(data_dict, phase="train")
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self.lr_decay()


    def forward_backward(self, data_dict, phase: str = "train"):

        if self.recursive_flag == 0:
            self.batch_image_input = data_dict.pop('input')
            self.batch_image_seg = data_dict.pop('seg')
            self.brain_mask = data_dict.pop('brainmask')
            self.gauss_mask = data_dict.pop('gauss_mask')
            self.batch_image_input = self.batch_image_input.to(self.device)
            self.batch_image_seg = self.batch_image_seg.to(self.device)
            self.brain_mask = self.brain_mask.to(self.device)
            self.gauss_mask = self.gauss_mask.to(self.device)

        assert phase in ["train", "val"]

        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        self.mp_trainer.zero_grad()

        losses = self.training_losses(
            self.model,
            self.batch_image_input,
            self.gauss_mask,
            self.brain_mask,
            self.args.model_name)
        loss = losses["loss"].mean()

        if phase == "train":
            self.mp_trainer.backward(loss)

        '''plot training loss'''
        self.writer.add_scalar('Loss', loss.detach().cpu().numpy(), self.step)


    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def lr_decay(self):
        if self.lr_decay_steps == 0 or self.step % self.lr_decay_steps != 0 or self.step == 0:
            return
        print('lr decay.....')
        n_decays = self.step // self.lr_decay_steps
        lr = self.lr * self.lr_decay_factor ** n_decays
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1))
        logger.logkv("time 100iter", self.time_iter)

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log(f"saving model ...")
            filename = f"model{(self.step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

    def training_losses(self, model, input_img, input_mask, brain_mask, model_name):
        """
        Compute the loss for the model given the input.

        Args:
            model: The model to compute the loss for.
            input_img: The input image.
            input_mask: The random mask used for training.
            brain_mask: The mask for the brain.
            model_name: The name of the model to train. 
        """
        terms = {}

        if model_name == 'masked_autoencoder':
            ### high freqency condition ###
            y_input = fft.fftshift(fft.fft2(input_img))
            center = (input_img.shape[2] // 2, input_img.shape[3] // 2)
            X, Y = np.ogrid[:input_img.shape[2], :input_img.shape[3]]
            radius = 15
            dist_from_center1 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            mask2 = th.from_numpy((dist_from_center1 >= radius)).cuda()
            mask = mask2
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

            noise = th.randn_like(input_img)
            input_mask = (1 - input_mask) * brain_mask
            input = (1 - input_mask) * input_img + input_mask * noise
            input = th.cat((input, x_cond), 1)
            x_start_pred = model(input.float())

            target = input_img
            terms["loss"] = mean_flat((target - x_start_pred) ** 2)

        elif model_name == 'first_iter':
            ### high freqency condition ###
            y_input = fft.fftshift(fft.fft2(input_img))
            center = (input_img.shape[2] // 2, input_img.shape[3] // 2)
            X, Y = np.ogrid[:input_img.shape[2], :input_img.shape[3]]
            radius = 15
            dist_from_center1 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            mask2 = th.from_numpy((dist_from_center1 < radius)).cuda()
            mask = mask2
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

            input = x_mask_real
            x_start_pred = model(input)
            target = input_img
            terms["loss"] = mean_flat((target - x_start_pred) ** 2)

        return terms


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()



def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None
