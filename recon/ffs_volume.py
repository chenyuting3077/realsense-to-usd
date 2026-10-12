"""ONNX-compatible cost volumes for the FFS export, the same values as FFS's scripts/make_single_onnx.py
ones (up to rounding) with less work: the target shifted by one Gather (not maxdisp pads stacked), the
group-wise correlation as matrix products with each feature vector normalized once (not once per
disparity: a vector shifted in from the padding is zero either way). At 848x480 the engine takes
127 -> 111 ms (TF32) for the Gather and the single normalization, 74 -> 62 ms (fp16 autocast) for the
matrix products (docs/profile)."""
import torch
import torch.nn.functional as F


def shifted(target, maxdisp):
    """BxCxHxW -> BxCxDxHxW, [:, :, d, :, w] = target[..., w - d] (0 for w < d)."""
    B, C, H, W = target.shape
    padded = F.pad(target, (maxdisp - 1, 0, 0, 0))
    w, d = torch.arange(W), torch.arange(maxdisp)  # on the CPU: the ONNX export folds it to a constant
    idx = (w[None, :] - d[:, None] + maxdisp - 1).reshape(-1).to(target.device)  # D*W into the padded width
    return padded.index_select(3, idx).view(B, C, H, maxdisp, W).permute(0, 1, 3, 2, 4)


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, normalize=True):
    """Group-wise correlation as matrix products: per group and image row, every left pixel with every
    right pixel (W x W, on the tensor cores), then the maxdisp diagonals w' = w - d picked by one Gather.
    No BxGxCxDxHxW product is built (6x the data at 848x480). Each feature vector is normalized once."""
    dtype = refimg_fea.dtype
    B, C, H, W = refimg_fea.shape
    ref = refimg_fea.view(B, num_groups, C // num_groups, H, W)
    tgt = targetimg_fea.view(B, num_groups, C // num_groups, H, W)
    if normalize:
        ref = F.normalize(ref.float(), dim=2).to(dtype)
        tgt = F.normalize(tgt.float(), dim=2).to(dtype)
    m = ref.permute(0, 1, 3, 4, 2) @ tgt.permute(0, 1, 3, 2, 4)  # BxGxHxWxW: [.., w, w'] = <ref(w), tgt(w')>
    m = F.pad(m, (maxdisp - 1, 0)).reshape(B, num_groups, H, W * (W + maxdisp - 1))  # w' < 0: 0
    w, d = torch.arange(W), torch.arange(maxdisp)  # on the CPU: the ONNX export folds it to a constant
    idx = (w[:, None] * (W + maxdisp - 1) + w[:, None] - d[None, :] + maxdisp - 1).reshape(-1).to(m.device)
    return m.index_select(3, idx).view(B, num_groups, H, W, maxdisp).permute(0, 1, 4, 2, 3).contiguous()


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    ref = refimg_fea.unsqueeze(2).expand(B, C, maxdisp, H, W)
    return torch.cat((ref, shifted(targetimg_fea, maxdisp)), dim=1).contiguous()
