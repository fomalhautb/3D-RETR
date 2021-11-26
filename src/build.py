from .models import VoxelDecoderMLP, VoxelDecoderCNN


def build_voxel_decoder(
        decoder_model='vae',
        decoder_dropout=0.3,
        decoder_depth=6,
        decoder_heads=6,
        decoder_dim=768,
        patch_num=4,
        voxel_size=32,
        num_resnet_blocks=2,
        cnn_hidden_dim=64,
        num_cnn_layers=3
):
    if decoder_model == 'mlp':
        return VoxelDecoderMLP(
            attn_dropout=decoder_dropout,
            depth=decoder_depth,
            heads=decoder_heads,
            dim=decoder_dim,
            patch_num=patch_num,
            voxel_size=voxel_size
        )
    elif decoder_model == 'cnn':
        return VoxelDecoderCNN(
            attn_dropout=decoder_dropout,
            depth=decoder_depth,
            heads=decoder_heads,
            dim=decoder_dim,
            patch_num=patch_num,
            voxel_size=voxel_size,
            num_resnet_blocks=num_resnet_blocks,
            cnn_hidden_dim=cnn_hidden_dim,
            num_cnn_layers=num_cnn_layers
        )
    else:
        raise ValueError('Unsupported decoder type')
