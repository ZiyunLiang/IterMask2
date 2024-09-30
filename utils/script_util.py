import ml_collections
from models import unet

def create_model(config: ml_collections.ConfigDict, image_level_cond):
    return unet.UNetModel(
        in_channels=config.model.num_input_channels,
        model_channels=config.model.num_channels,
        out_channels=config.model.num_input_channels,
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=tuple(config.model.attention_ds),
        dropout=config.model.dropout,
        channel_mult=config.model.channel_mult,
        dims=config.model.dims,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=config.model.num_heads,
        num_head_channels=config.model.num_head_channels,
        num_heads_upsample=config.model.num_heads_upsample,
        use_scale_shift_norm=config.model.use_scale_shift_norm,
        resblock_updown=config.model.resblock_updown,
        image_level_cond=image_level_cond,
    )