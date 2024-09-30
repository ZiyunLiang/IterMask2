import ml_collections

import torch as th

def get_default_configs():
    config = ml_collections.ConfigDict()
    config.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    config.seed = 1
    config.data = data = ml_collections.ConfigDict()
    data.path = "/home/some5338/Documents/data/brats"
    data.sequence_translation = False
    data.healthy_data_percentage = None

    ## model config
    config.model = model = ml_collections.ConfigDict()
    model.image_size = 128
    model.num_input_channels = 1 
    model.num_channels = 32
    model.num_res_blocks = 2
    model.num_heads = 1
    model.num_heads_upsample = -1
    model.num_head_channels = -1
    model.attention_resolutions = "32,16,8"

    attention_ds = []
    if model.attention_resolutions != "":
        for res in model.attention_resolutions.split(","):
            attention_ds.append(model.image_size // int(res))
    model.attention_ds = attention_ds

    model.channel_mult = {64:(1, 2, 3, 4), 128:(1, 1, 2, 3, 4)}[model.image_size]
    model.dropout = 0.1
    model.use_checkpoint = False
    model.use_scale_shift_norm = True
    model.resblock_updown = True
    model.use_fp16 = False
    model.use_new_attention_order = False
    model.dims = 2

    # score model training
    config.model.training = training_model = ml_collections.ConfigDict()
    training_model.lr = 1e-4
    training_model.weight_decay = 0.00
    training_model.lr_decay_steps = 150000
    training_model.lr_decay_factor = 0.1
    training_model.batch_size = 32
    training_model.ema_rate = "0.9999"  # comma-separated list of EMA values
    training_model.log_interval = 100
    training_model.save_interval = 5000
    training_model.use_fp16 = model.use_fp16
    training_model.fp16_scale_growth = 1e-3
    training_model.iterations = 150000

    config.testing = testing = ml_collections.ConfigDict()
    testing.batch_size = 32
    testing.task = 'inpainting'

    return config