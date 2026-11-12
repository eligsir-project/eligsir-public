import torch

from .modules.lpips import LPIPS


# Cache LPIPS networks by (net_type, version, device). Rebuilding the VGG/Alex
# backbone (instantiation + weight load + .to(device)) on every call dominated
# evaluation time — a per-view metric was reloading a full network hundreds of
# times per report. Build once, reuse.
_LPIPS_CACHE = {}


def _get_lpips(net_type: str, version: str, device: torch.device) -> LPIPS:
    key = (net_type, version, str(device))
    criterion = _LPIPS_CACHE.get(key)
    if criterion is None:
        criterion = LPIPS(net_type, version).to(device).eval()
        _LPIPS_CACHE[key] = criterion
    return criterion


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.

    The backbone network is cached per (net_type, version, device) so repeated
    metric calls (e.g. one per evaluation view) do not rebuild it. Runs under
    no_grad since LPIPS is only used as a metric here.
    """
    criterion = _get_lpips(net_type, version, x.device)
    with torch.no_grad():
        return criterion(x, y)
