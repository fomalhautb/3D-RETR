# based on https://github.com/lucidrains/x-transformers

from functools import partial

from torch import nn

from .attention import Attention
from .modules import (
    groupby_prefix_and_trim,
    equals,
    Residual,
    FeedForward,
    LayerIntermediates
)


class AttentionLayers(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            heads: int,
            causal: bool = False,
            cross_attend: bool = False,
            only_cross: bool = False,
            **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        norm_fn = partial(nn.LayerNorm, dim)

        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        for layer_type in self.layer_types:
            if layer_type == 'a':
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            residual_fn = Residual()

            self.layers.append(nn.ModuleList([
                norm_fn(),
                layer,
                residual_fn
            ]))

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None,
            return_hiddens=False
    ):
        hiddens = []
        intermediates = []

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == 'a':
                hiddens.append(x)

            residual = x
            x = norm(x)

            if layer_type == 'a':
                out, inter = block(x, mask=mask)
            elif layer_type == 'c':
                out, inter = block(x, context=context, mask=mask, context_mask=context_mask)
            elif layer_type == 'f':
                out = block(x)

            x = residual_fn(out, residual)

            if layer_type in ('a', 'c'):
                intermediates.append(inter)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens,
                attn_intermediates=intermediates
            )

            return x, intermediates

        return x
