import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .losses import DiceLoss, CEDiceLoss, FocalLoss
from .transformer import AttentionLayers


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class VoxelDecoderMLP(nn.Module):
    def __init__(
            self,
            patch_num: int = 4,
            voxel_size: int = 32,
            dim: int = 512,
            depth: int = 6,
            heads: int = 8,
            dim_head: int = 64,
            attn_dropout: float = 0.0,
            ff_dropout: float = 0.0,
    ):
        super().__init__()

        if voxel_size % patch_num != 0:
            raise ValueError('voxel_size must be dividable by patch_num')

        self.patch_num = patch_num
        self.voxel_size = voxel_size
        self.patch_size = voxel_size // patch_num
        self.emb = nn.Embedding(patch_num ** 3, dim)
        self.transformer = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=False,
            cross_attend=True
        )

        self.layer_norm = nn.LayerNorm(dim)
        self.to_patch = nn.Linear(dim, self.patch_size ** 3)

    def generate(
            self,
            context: Tensor,
            context_mask: Tensor = None,
            **kwargs
    ):
        out = self(context, context_mask)
        return torch.sigmoid(out)

    def forward(
            self,
            context: Tensor,
            context_mask: Tensor = None
    ) -> Tensor:
        x = self.emb(torch.arange(self.patch_num ** 3, device=context.device))
        x = x.unsqueeze(0).repeat(context.shape[0], 1, 1)
        out = self.transformer(x=x, context=context, context_mask=context_mask)
        out = self.layer_norm(out)
        out = out.view(self.patch_num ** 3 * context.shape[0], -1)
        patched = self.to_patch(out)

        return patched.view(-1, self.voxel_size, self.voxel_size, self.voxel_size)

    def get_loss(
            self,
            x: Tensor,
            context: Tensor,
            context_mask: Tensor = None,
            **kwargs
    ):
        out = self(context, context_mask)
        return F.binary_cross_entropy_with_logits(
            out.view(out.size(0), -1),
            x.view(out.size(0), -1)
        )


class VoxelDecoderCNN(nn.Module):
    def __init__(
            self,
            patch_num: int = 4,
            num_cnn_layers: int = 3,
            num_resnet_blocks: int = 2,
            cnn_hidden_dim: int = 64,
            voxel_size: int = 32,
            dim: int = 512,
            depth: int = 6,
            heads: int = 8,
            dim_head: int = 64,
            attn_dropout: float = 0.0,
            ff_dropout: float = 0.0,
    ):
        super().__init__()

        if voxel_size % patch_num != 0:
            raise ValueError('voxel_size must be dividable by patch_num')

        self.patch_num = patch_num
        self.voxel_size = voxel_size
        self.patch_size = voxel_size // patch_num
        self.emb = nn.Embedding(patch_num ** 3, dim)
        self.transformer = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=False,
            cross_attend=True
        )

        has_resblocks = num_resnet_blocks > 0
        dec_chans = [cnn_hidden_dim] * num_cnn_layers
        dec_init_chan = dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        dec_chans_io = list(zip(dec_chans[:-1], dec_chans[1:]))

        dec_layers = []

        for (dec_in, dec_out) in dec_chans_io:
            dec_layers.append(nn.Sequential(nn.ConvTranspose3d(dec_in, dec_out, 4, stride=2, padding=1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv3d(dim, dec_chans[1], 1))

        dec_layers.append(nn.Conv3d(dec_chans[-1], 1, 1))

        self.decoder = nn.Sequential(*dec_layers)

        self.layer_norm = nn.LayerNorm(dim)
        self.to_patch = nn.Linear(dim, self.patch_size ** 3)

    def generate(
            self,
            context: Tensor,
            context_mask: Tensor = None,
            **kwargs
    ):
        out = self(context, context_mask)
        return torch.sigmoid(out)

    def forward(
            self,
            context: Tensor,
            context_mask: Tensor = None
    ) -> Tensor:
        x = self.emb(torch.arange(self.patch_num ** 3, device=context.device))
        x = x.unsqueeze(0).repeat(context.shape[0], 1, 1)
        out = self.transformer(x=x, context=context, context_mask=context_mask)
        out = self.layer_norm(out)
        out = rearrange(out, 'b (h w c) d -> b d h w c', h=self.patch_num, w=self.patch_num, c=self.patch_num)
        out = self.decoder(out)

        return out

    def get_loss(
            self,
            x: Tensor,
            context: Tensor,
            context_mask: Tensor = None,
            loss_type='dice'
    ):
        out = self(context, context_mask)
        out = out.view(out.size(0), -1)
        x = x.view(out.size(0), -1)

        if loss_type == 'ce':
            loss_fn = F.binary_cross_entropy_with_logits
        elif loss_type == 'dice':
            loss_fn = DiceLoss()
        elif loss_type == 'ce_dice':
            loss_fn = CEDiceLoss()
        elif loss_type == 'focal':
            loss_fn = FocalLoss()
        else:
            raise ValueError(f'Unsupported loss type "{loss_type}"')

        return loss_fn(out, x)
