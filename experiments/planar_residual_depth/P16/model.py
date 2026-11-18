"""Pixel-wise residual refinement and differentiable camera reprojection."""
import torch
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(8, outputs), nn.SiLU(),
        nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(8, outputs), nn.SiLU(),
    )


class Refiner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        widths = (24, 48, 96, 128)
        self.encoders = nn.ModuleList(block(a, b) for a, b in zip((channels, *widths), widths))
        self.decoders = nn.ModuleList(block(a+b, b) for a,b in zip(widths[:0:-1], widths[-2::-1]))
        self.head = nn.Conv2d(24, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs, raw, editable):
        features = []
        for encoder in self.encoders:
            if features:
                inputs = F.max_pool2d(inputs, 2)
            inputs = encoder(inputs)
            features.append(inputs)
        for decoder, skip in zip(self.decoders, features[-2::-1]):
            inputs = F.interpolate(inputs, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            inputs = decoder(torch.cat([inputs, skip], dim=1))
        gate, residual = self.head(inputs).float().chunk(2, dim=1)
        gate = gate.sigmoid() * editable
        output = raw * (1 + gate * .3 * residual.tanh())
        return output, gate


def warp(depth, ref_k, src_k, relative, source):
    """Sample the source raster, using camera-Z and world-to-camera poses."""
    b, _, h, w = depth.shape
    sy, sx = source.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h, device=depth.device),
                          torch.arange(w, device=depth.device), indexing='ij')
    pixels = torch.stack([x, y, torch.ones_like(x)]).reshape(3, -1).float()
    xyz = torch.linalg.solve(ref_k, pixels.expand(b,-1,-1)) * depth.flatten(2)
    xyz = relative[:, :3, :3] @ xyz + relative[:, :3, 3:4]
    z = xyz[:, 2:3]
    projected = src_k @ xyz
    uv = projected[:, :2] / z.clamp_min(1e-6)
    grid = (uv.transpose(1, 2).reshape(b,h,w,2) /
            depth.new_tensor([max(1,sx-1),max(1,sy-1)]))*2-1
    valid = (z.reshape(b,1,h,w)>1e-6) & (grid.abs() <= 1.000001).all(-1).unsqueeze(1)
    sampled = F.grid_sample(source, grid.clamp(-1,1), align_corners=True, padding_mode='border')
    return sampled, valid, z.reshape(b,1,h,w)


def rgb_error(ref, other):
    def avg(image):
        return F.avg_pool2d(F.pad(image, (1,1,1,1),mode='replicate'),3,1)
    a,b = avg(ref),avg(other)
    va,vb = avg(ref.square())-a.square(),avg(other.square())-b.square()
    cov = avg(ref*other)-a*b
    ssim = ((2*a*b+.01**2)*(2*cov+.03**2) /
            ((a.square()+b.square()+.01**2)*(va.clamp_min(0)+vb.clamp_min(0)+.03**2)))
    return (.15*(ref-other).abs()+.85*((1-ssim)/2).clamp(0,1)).mean(1, keepdim=True)


def evidence(batch, camera, corrected=None, volume=False):
    """Freeze visibility on raw geometry and penalize leaving the source raster."""
    raw = batch['raw']
    ref = batch['rgb']
    k, pose, source_k, source_pose = (batch[f'{camera}_{key}'] for key in
                                         ('K','pose','source_K','source_pose'))
    masks, errors, inside_flags = [], [], []
    for j in range(batch['source_rgb'].shape[1]):
        relative = source_pose[:,j] @ torch.linalg.inv(pose)
        raw_z = batch['source_raw'][:,j]
        source = torch.cat([batch['source_rgb'][:,j],raw_z,batch['source_static'][:,j].float()],1)
        _,_,h,w = raw.shape
        with torch.no_grad():
            sampled, inside, z = warp(raw,k,source_k[:,j],relative,source)
            # Relative translation provides a lower bound on useful baseline;
            # per-pixel transverse displacement is checked separately.
            rotation_only = relative.clone()
            rotation_only[:,:3,3] = 0
            # Use a pixel-coordinate raster to measure parallax after removing rotation.
            yy,xx = torch.meshgrid(torch.arange(h,device=raw.device),torch.arange(w,device=raw.device),indexing='ij')
            coords = torch.stack([xx,yy]).float().expand(len(raw),-1,-1,-1)
            moved,_,_ = warp(raw,k,source_k[:,j],relative,coords)
            rotated,_,_ = warp(raw,k,source_k[:,j],rotation_only,coords)
            parallax = (moved-rotated).square().sum(1,keepdim=True).sqrt()
            visible = (inside & (sampled[:,4:5]>.999) & batch['static'] &
                       (sampled[:,3:4]>0) & ((sampled[:,3:4]-z).abs()<.08*z) &
                       (parallax>=.75) & batch['has_source'][:,j,None,None,None])
            masks.append(visible)
        if volume:
            channels=[]
            for step in torch.linspace(-.3,.3,9,device=raw.device):
                sample, in_bounds, _ = warp(raw*(1+step), k,source_k[:,j],relative,source[:,:3])
                channels.append(torch.where(visible & in_bounds,rgb_error(ref,sample),2))
            errors.append(torch.cat(channels,1))
        else:
            query = corrected if corrected is not None else raw
            sample, in_bounds, _ = warp(query,k,source_k[:,j],relative,source[:,:3])
            errors.append(torch.where(visible,torch.where(in_bounds,rgb_error(ref,sample),1),2))
            inside_flags.append(visible & in_bounds)
    visible = torch.stack(masks).any(0)
    cost = torch.stack(errors).amin(0)
    if volume:
        return cost
    return cost,visible,torch.stack(inside_flags).any(0)
