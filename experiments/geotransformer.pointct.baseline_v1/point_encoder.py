from typing import Optional

import torch
import torch.nn as nn


class PointEncoderContractError(RuntimeError):
    pass


class PointEncoder(nn.Module):
    """M2-1 KPConv Point encoder with explicit network/physical coordinates."""

    def __init__(self, cfg, backbone: Optional[nn.Module] = None):
        super().__init__()
        point_cfg = cfg.point
        self.point_network_scale_mm_to_m = float(point_cfg.point_network_scale_mm_to_m)
        self.coarse_raw_dim = int(point_cfg.coarse_raw_dim)
        self.projected_dim = int(point_cfg.projected_dim)
        if self.point_network_scale_mm_to_m != 0.001:
            raise PointEncoderContractError('M2-1 point_network_scale_mm_to_m must remain frozen at 0.001.')

        if backbone is None:
            # Standard local experiment import; the compiled extension is only
            # required when constructing the real KPConv backbone.
            from backbone import KPConvFPN

            backbone = KPConvFPN(
                input_dim=point_cfg.input_dim,
                output_dim=point_cfg.fine_feature_dim,
                init_dim=point_cfg.init_dim,
                kernel_size=point_cfg.kernel_size,
                init_radius=point_cfg.init_radius_m,
                init_sigma=point_cfg.init_sigma_m,
                group_norm=point_cfg.group_norm,
            )
        self.backbone = backbone
        self.point_proj = nn.Linear(self.coarse_raw_dim, self.projected_dim)

    @staticmethod
    def _require_finite(name, tensor):
        if not torch.is_tensor(tensor):
            raise PointEncoderContractError(f'{name} must be a torch.Tensor.')
        if not torch.isfinite(tensor).all():
            raise PointEncoderContractError(f'{name} contains NaN or Inf.')

    def forward(self, point_dict):
        """Encode only ``data_dict['point']`` from the dual-branch batch."""
        if 'features' not in point_dict:
            raise PointEncoderContractError('Point encoder input is missing features.')
        if 'points' not in point_dict or len(point_dict['points']) != 4:
            raise PointEncoderContractError('Point encoder requires exactly four KPConv point stages.')
        features = point_dict['features']
        if not torch.is_tensor(features) or features.ndim != 2 or features.shape[1] != 1:
            raise PointEncoderContractError('Baseline V1 Point features must have shape [N,1].')
        if features.dtype != torch.float32 or not torch.equal(features, torch.ones_like(features)):
            raise PointEncoderContractError('Baseline V1 Point features must be float32 all-ones values.')
        input_scale = float(point_dict.get('point_network_scale_mm_to_m', -1.0))
        if input_scale != self.point_network_scale_mm_to_m:
            raise PointEncoderContractError('Point preprocessing/encoder network scales are inconsistent.')

        feats_list = self.backbone(features, point_dict)
        if len(feats_list) != 3:
            raise PointEncoderContractError(f'KPConvFPN must return three feature levels; got {len(feats_list)}.')

        # KPConvFPN is ordered fine-to-coarse. Projection is intentionally
        # applied exactly once and no M3 L2 normalization happens here.
        P_raw = feats_list[-1]
        Xp_net_coarse = point_dict['points'][3]
        Xp_phys_coarse = Xp_net_coarse / self.point_network_scale_mm_to_m
        Q = self.point_proj(P_raw)

        for name, tensor in (
            ('P_raw', P_raw),
            ('Q', Q),
            ('Xp_net_coarse', Xp_net_coarse),
            ('Xp_phys_coarse', Xp_phys_coarse),
        ):
            self._require_finite(name, tensor)

        if P_raw.ndim != 2 or P_raw.shape[1] != self.coarse_raw_dim:
            raise PointEncoderContractError(
                f'P_raw must have shape [Np,{self.coarse_raw_dim}]; got {tuple(P_raw.shape)}.'
            )
        if P_raw.shape[0] == 0:
            raise PointEncoderContractError('P_raw must contain at least one coarse Point token.')
        if Q.ndim != 2 or Q.shape[1] != self.projected_dim:
            raise PointEncoderContractError(
                f'Q must have shape [Np,{self.projected_dim}]; got {tuple(Q.shape)}.'
            )
        if Xp_net_coarse.ndim != 2 or Xp_net_coarse.shape[1] != 3:
            raise PointEncoderContractError(
                f'Xp_net_coarse must have shape [Np,3]; got {tuple(Xp_net_coarse.shape)}.'
            )
        if Xp_phys_coarse.ndim != 2 or Xp_phys_coarse.shape[1] != 3:
            raise PointEncoderContractError(
                f'Xp_phys_coarse must have shape [Np,3]; got {tuple(Xp_phys_coarse.shape)}.'
            )

        token_counts = {
            P_raw.shape[0],
            Q.shape[0],
            Xp_net_coarse.shape[0],
            Xp_phys_coarse.shape[0],
        }
        if len(token_counts) != 1:
            raise PointEncoderContractError('Point feature and coordinate token counts are inconsistent.')
        if not torch.allclose(
            Xp_phys_coarse * self.point_network_scale_mm_to_m,
            Xp_net_coarse,
            rtol=1e-5,
            atol=1e-7,
        ):
            raise PointEncoderContractError('Coarse physical/network coordinate scaling is inconsistent.')

        return {
            'P_raw': P_raw,
            'Q': Q,
            'Xp_net_coarse': Xp_net_coarse,
            'Xp_phys_coarse': Xp_phys_coarse,
        }


def create_point_encoder(cfg):
    return PointEncoder(cfg)


__all__ = ['PointEncoder', 'PointEncoderContractError', 'create_point_encoder']
