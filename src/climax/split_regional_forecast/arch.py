# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from climax.my_arch3 import ClimaX3

class SplitRegionalClimaX(ClimaX3):
    def __init__(self, default_vars, img_size=..., patch_size=2, embed_dim=1024, depth=8, decoder_depth=2, num_heads=16, num_representative=2, mlp_ratio=4, drop_path=0.1, drop_rate=0.1, parallel_patch_embed=False, delete_pexel_num=0, is_used_pos_embed=None):
        super().__init__(default_vars, img_size, patch_size, embed_dim, depth, decoder_depth, num_heads, num_representative, mlp_ratio, drop_path, drop_rate, parallel_patch_embed, is_used_pos_embed)
        self.delete_pexel_num = delete_pexel_num
        self.is_used_pos_embed = is_used_pos_embed

    def forward_encoder(self, x0: torch.Tensor, lead_times: torch.Tensor, variables, region_info):
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

            # get the patch ids corresponding to the region
            region_patch_ids = region_info['patch_ids']
            x = x[:, :, region_patch_ids, :]

            # variable aggregation
            x = self.aggregate_variables(x, k)  # B, L, D

            # add pos embedding
            if self.is_used_pos_embed:
                x = x + self.pos_embed[k][:, region_patch_ids, :]

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
            xs = self.blocks[i](xs)
            xs = self.norm[i](xs)
            if i < self.depth - 1:
                xo = self.cross_over[i]
                xs = torch.einsum("rbld->blrd", xs) # R, B, L, D
                xs = torch.reshape(xs, (-1, *xs.shape[2:])) # (B, L), R, D
                xs = xo(xs) # (B, L), R, D
                xs = torch.reshape(xs, (batch_size, -1, *xs.shape[1:])) # B, L, R, D
                xs = torch.einsum("blrd->rbld", xs)
        return xs

    def forward(self, x, y, lead_times, variables, out_variables, metric, lat, region_info):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.
            region_info: Containing the region's information

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        out_transformers = self.forward_encoder(x, lead_times, variables, region_info)  # B, L, D

        preds = []
        for head, output in zip(self.heads, out_transformers):
            preds.append(head(output))
        preds = sum(preds)/len(preds)  # B, L, V*p*p

        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        preds = self.unpatchify(preds, h = max_h - min_h + 1, w = max_w - min_w + 1)
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        if self.delete_pexel_num != 0:
            preds = preds[:, out_var_ids, self.delete_pexel_num:-self.delete_pexel_num, self.delete_pexel_num:-self.delete_pexel_num]
        else:
            preds = preds[:, out_var_ids]

        y = y[:, :, min_h+self.delete_pexel_num:max_h+1-self.delete_pexel_num, min_w+self.delete_pexel_num:max_w+1-self.delete_pexel_num]
        lat = lat[min_h+self.delete_pexel_num:max_h+1-self.delete_pexel_num]

        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]

        return loss, preds

    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, clim, log_postfix, region_info):
        _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat, region_info=region_info)

        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        y = y[:, :, min_h+self.delete_pexel_num:max_h+1-self.delete_pexel_num, min_w+self.delete_pexel_num:max_w+1-self.delete_pexel_num]
        lat = lat[min_h+self.delete_pexel_num:max_h+1-self.delete_pexel_num]
        clim = clim[:, min_h+self.delete_pexel_num:max_h+1-self.delete_pexel_num, min_w+self.delete_pexel_num:max_w+1-self.delete_pexel_num]

        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]
