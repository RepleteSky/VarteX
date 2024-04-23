# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_

from climax.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)

from .parallelpatchembed import ParallelVarPatchEmbed


class ClimaX2(nn.Module):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
        parallel_patch_embed (bool): whether to use parallel patch embedding
    """

    def __init__(
        self,
        default_vars,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        num_representative=8,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
    ):
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        self.parallel_patch_embed = parallel_patch_embed
        self.num_heads = num_heads
        self.num_representatives = num_representative
        self.representative_dim = embed_dim // self.num_representatives
        self.depth = depth
        # variable tokenization: separate embedding layer for each input variable
        if self.parallel_patch_embed:
            self.token_embeds = ParallelVarPatchEmbed(len(default_vars), img_size, patch_size, self.representative_dim)
            self.num_patches = self.token_embeds.num_patches
        else:
            self.token_embeds = nn.ModuleList([nn.ModuleList(
                [PatchEmbed(img_size, patch_size, 1, self.representative_dim) for i in range(len(default_vars))]
            ) for _ in range(self.num_representatives)])
            self.num_patches = self.token_embeds[0][0].num_patches

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        _, self.var_map = self.create_var_embedding(self.representative_dim)
        self.var_embed = nn.ParameterList([self.create_var_embedding(self.representative_dim)[0] for _ in range(self.num_representatives)])

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, self.representative_dim), requires_grad=True) for _ in range(self.num_representatives)])
        self.var_agg = nn.ModuleList([nn.MultiheadAttention(self.representative_dim, self.num_heads, batch_first=True) for _ in range(self.num_representatives)])

        # positional embedding and lead time embedding
        self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, self.num_patches, self.representative_dim),requires_grad=True)
                                           for _ in range(self.num_representatives)])
        self.lead_time_embed = nn.Linear(1, self.representative_dim)

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            encoders = nn.ModuleList(
                [
                    Block(
                        self.representative_dim,
                        self.num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        drop_path=dpr[i],
                        norm_layer=nn.LayerNorm,
                        drop=drop_rate,
                    )
                    for i in range(self.num_representatives)
                ]
            )
            self.blocks.append(encoders)
        self.norm = nn.ModuleList([nn.LayerNorm(self.representative_dim) for _ in range(self.num_representatives)])

        # Cross over
        self.cross_over = nn.ModuleList(
            [
                Block(
                    self.representative_dim,
                    self.num_heads//self.num_representatives,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                )
                for i in range(depth-1)
            ]
        )


        # --------------------------------------------------------------------------

        # prediction head
        self.heads = nn.ModuleList()
        for _ in range(self.num_representatives):
            head = nn.ModuleList()
            for _ in range(decoder_depth):
                head.append(nn.Linear(self.representative_dim, self.representative_dim))
                head.append(nn.GELU())
            head.append(nn.Linear(self.representative_dim, len(self.default_vars) * patch_size**2))
            head = nn.Sequential(*head)
            self.heads.append(head)

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed[0].shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed[0].shape[-1], np.arange(len(self.default_vars)))

        for k in range(self.num_representatives):
            self.pos_embed[k].data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            self.var_embed[k].data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

            # token embedding layer
            if self.parallel_patch_embed:
                for i in range(len(self.token_embeds.proj_weights)):
                    w = self.token_embeds.proj_weights[k][i].data # パラレルパッチのKは未確認
                    trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
            else:
                for i in range(len(self.token_embeds)):
                    w = self.token_embeds[k][i].proj.weight.data
                    trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = len(self.default_vars)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_variables(self, x: torch.Tensor, k: int):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query[k].repeat_interleave(x.shape[0], dim=0) # BxL, 1, D
        x, _ = self.var_agg[k](var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x


    def forward_encoder(self, x0: torch.Tensor, lead_times: torch.Tensor, variables):
        # x: `[B, V, H, W]` shape.

        if isinstance(variables, list):
            variables = tuple(variables)

        xs = []
        for k in range(self.num_representatives):
            # tokenize each variable separately
            embeds = []
            var_ids = self.get_var_ids(variables, x0.device)

            if self.parallel_patch_embed:
                x = self.token_embeds(x0, var_ids)  # B, V, L, D
            else:
                for i in range(len(var_ids)):
                    id = var_ids[i]
                    embeds.append(self.token_embeds[k][id](x0[:, i : i + 1]))
                x = torch.stack(embeds, dim=1)  # B, V, L, D

            # add variable embedding
            var_embed = self.get_var_emb(self.var_embed[k], variables)
            x = x + var_embed.unsqueeze(2)  # B, V, L, D

            # variable aggregation
            x = self.aggregate_variables(x, k)  # B, L, D

            # add pos embedding
            x = x + self.pos_embed[k]

            # add lead time embedding
            lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
            lead_time_emb = lead_time_emb.unsqueeze(1)
            x = x + lead_time_emb  # B, L, D

            x = self.pos_drop(x)

            xs.append(x.unsqueeze(0)) # R, B, L, D

        xs = torch.cat(xs, dim=0) # R, B, L, D

        batch_size = xs.shape[1]
        # apply Transformer blocks
        for i in range(self.depth):
            blk = self.blocks[i]
            xs_list = []
            for k, (enc, ln) in enumerate(zip(blk, self.norm)):
                z = enc(xs[k]) # B, L, D
                z = ln(z)
                xs_list.append(z.unsqueeze(0)) # 1, B, L, D
            xs = torch.cat(xs_list, dim=0)
            if i < self.depth - 1:
                xo = self.cross_over[i]
                xs = torch.einsum("rbld->blrd", xs) # R, B, L, D
                xs = torch.reshape(xs, (-1, *xs.shape[2:])) # (B, L), R, D
                xs = xo(xs) # (B, L), R, D
                xs = torch.reshape(xs, (batch_size, -1, *xs.shape[1:])) # B, L, R, D
                xs = torch.einsum("blrd->rbld", xs)
        return xs

    def forward(self, x, y, lead_times, variables, out_variables, metric, lat):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        out_transformers = self.forward_encoder(x, lead_times, variables)  # R, B, L, D

        preds = []
        for head, output in zip(self.heads, out_transformers):
            preds.append(head(output))
        preds = sum(preds)/len(preds)  # B, L, V*p*p

        preds = self.unpatchify(preds)
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        preds = preds[:, out_var_ids]

        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]

        return loss, preds

    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, clim, log_postfix):
        _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat)
        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]
