"""Model factory + Grad-CAM that works for training (alignment loss) and evaluation."""
import timm
import torch
import torch.nn.functional as F

from config import MODELS, NUM_CLASSES


def get_module(model, dotted):
    m = model
    for part in dotted.split("."):
        m = m[int(part)] if part.lstrip("-").isdigit() else getattr(m, part)
    return m


class CamModel(torch.nn.Module):
    """Wraps a timm classifier and records the target layer's feature map on every forward."""

    def __init__(self, name, pretrained=True):
        super().__init__()
        spec = MODELS[name]
        self.name = name
        self.net = timm.create_model(spec["timm"], pretrained=pretrained, num_classes=NUM_CLASSES)
        self.is_vit = name.startswith("vit")
        self.n_prefix = getattr(self.net, "num_prefix_tokens", 1) if self.is_vit else 0
        self._feat = None
        get_module(self.net, spec["layer"]).register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._feat = out

    def features(self):
        """Feature map as (B,C,h,w); ViT tokens are reshaped to a square grid."""
        f = self._feat
        if self.is_vit:
            f = f[:, self.n_prefix:, :]
            b, n, c = f.shape
            s = int(n ** 0.5)
            f = f.transpose(1, 2).reshape(b, c, s, s)
        return f

    def forward(self, x):
        return self.net(x)


def grad_cam(model: CamModel, logits, target, create_graph=False):
    """Grad-CAM for class `target` (B,) from the last forward pass.

    Channel weights are always detached (stop-gradient), so with create_graph=False the
    returned map is a detached evaluation map, and with create_graph=True the map keeps a
    graph through the feature map only (cheap: no second-order gradients).
    Returns (B,h,w) float32, non-negative.
    """
    raw = model._feat
    score = logits.gather(1, target.view(-1, 1)).sum()
    g = torch.autograd.grad(score, raw, retain_graph=True)[0]
    feat = model.features()
    if model.is_vit:
        g = g[:, model.n_prefix:, :]
        b, n, c = g.shape
        s = int(n ** 0.5)
        g = g.transpose(1, 2).reshape(b, c, s, s)
    w = g.float().mean(dim=(2, 3), keepdim=True).detach()
    f = feat.float() if create_graph else feat.float().detach()
    return F.relu((w * f).sum(dim=1))


def off_leaf_loss(cam, mask):
    """Share of CAM energy outside the leaf. cam (B,h,w), mask (B,1,H,W) -> scalar."""
    m = F.adaptive_avg_pool2d(mask, cam.shape[-2:]).squeeze(1)   # soft leaf fraction per cell
    tot = cam.sum(dim=(1, 2)) + 1e-6
    return ((cam * (1 - m)).sum(dim=(1, 2)) / tot).mean()


def lesion_focus_score(cam, mask, size):
    """LFS per image: min-max normalised CAM at full resolution, share of energy on the leaf."""
    c = F.interpolate(cam[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]
    lo = c.flatten(1).min(1)[0].view(-1, 1, 1)
    hi = c.flatten(1).max(1)[0].view(-1, 1, 1)
    c = (c - lo) / (hi - lo + 1e-8)
    m = mask[:, 0]
    return (c * m).sum(dim=(1, 2)) / (c.sum(dim=(1, 2)) + 1e-8), c
