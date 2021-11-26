from torch import nn

from .vision_transformer import (
    vit_deit_base_distilled_patch16_224,
    vit_deit_small_distilled_patch16_224,
    vit_deit_base_distilled_patch16_384,
    vit_deit_tiny_distilled_patch16_224,
    vit_large_patch16_224_in21k,
    vit_base_patch16_224_in21k
)


class VisionTransformerEncoder(nn.Module):
    def __init__(self, attn_dropout=0.0, model='vit_deit_base_distilled_patch16_224', pretrained=True):
        super(VisionTransformerEncoder, self).__init__()
        if model == 'vit_deit_tiny_distilled_patch16_224':
            self.model = vit_deit_tiny_distilled_patch16_224(attn_drop_rate=attn_dropout, pretrained=pretrained)
        elif model == 'vit_deit_base_distilled_patch16_224':
            self.model = vit_deit_base_distilled_patch16_224(attn_drop_rate=attn_dropout, pretrained=pretrained)
        elif model == 'vit_deit_small_distilled_patch16_224':
            self.model = vit_deit_small_distilled_patch16_224(attn_drop_rate=attn_dropout, pretrained=pretrained)
        elif model == 'vit_deit_base_distilled_patch16_384':
            self.model = vit_deit_base_distilled_patch16_384(attn_drop_rate=attn_dropout, pretrained=pretrained)
        elif model == 'vit_large_patch16_224_in21k':
            self.model = vit_large_patch16_224_in21k(attn_drop_rate=attn_dropout, pretrained=pretrained)
        elif model == 'vit_base_patch16_224_in21k':
            self.model = vit_base_patch16_224_in21k(attn_drop_rate=attn_dropout, pretrained=pretrained)
            raise ValueError('Unsupported model')

    def forward(self, x):
        return self.model(x)
