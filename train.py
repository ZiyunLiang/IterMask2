import time
import sys
import argparse
import os

from pathlib import Path
import torch as th

from configs import get_config
from utils import logger
from datasets import loader
from utils.script_util import create_model
from utils.train_util import TrainLoop
sys.path.append(str(Path.cwd()))


def main(args):
    use_gpus = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpus)

    time_load_start = time.time()
    config = get_config.file_from_dataset(args.dataset)

    if args.experiment_name != 'None':
        experiment_name = args.experiment_name
    else:
        experiment_name = args.model_name + '_' + args.dataset + '_' + args.modality

    logger.configure(Path(experiment_name),
                     format_strs=["log", "stdout", "csv", "tensorboard"])

    logger.log("creating model and diffusion...")
    if args.model_name == 'masked_autoencoder':
        image_level_cond = True
    elif args.model_name == 'first_iter':
        image_level_cond = False
    else:
        raise Exception("Model name does exit")

    model = create_model(config, image_level_cond)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    device = th.device(config.device)
    model.to(device)
    logger.log(f"Model number of parameters {pytorch_total_params}")
    input_modality = args.modality

    logger.log("creating data loader...")
    train_loader = loader.get_data_loader(args.dataset, args.data_dir, config, input_modality, split_set='train', generator=True)
    time_load_end = time.time()
    time_load = time_load_end - time_load_start
    logger.log("data loaded: time ", str(time_load))
    logger.log("training...")
    TrainLoop(
        model=model,
        data=train_loader,
        batch_size=config.model.training.batch_size,
        lr=config.model.training.lr,
        ema_rate=config.model.training.ema_rate,
        log_interval=config.model.training.log_interval,
        save_interval=config.model.training.save_interval,
        use_fp16=config.model.training.use_fp16,
        fp16_scale_growth=config.model.training.fp16_scale_growth,
        weight_decay=config.model.training.weight_decay,
        lr_decay_steps=config.model.training.lr_decay_steps,
        lr_decay_factor=config.model.training.lr_decay_factor,
        iterations=config.model.training.iterations,
        num_input_channels=config.model.num_input_channels,
        image_size=config.model.image_size,
        device=device,
        args=args
    ).run_loop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", help="the id of the gpu you want to use, like 0", type=int, default=0)
    parser.add_argument("--dataset", help="brats", type=str, default='brats')
    parser.add_argument("--modality", help="input modality, choose from flair, t2, t1", type=str, default='flair')
    parser.add_argument("--data_dir", help="data directory", type=str, default='./datasets/data')
    parser.add_argument("--experiment_name", help="model to load from", type=str, default='None')
    parser.add_argument("--model_name", help="which model to train: first_iter or masked_autoencoder", type=str, default='masked_autoencoder')
    args = parser.parse_args()
    main(args)


