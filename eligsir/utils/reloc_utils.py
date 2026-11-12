import torch

# DEBUG: force Python fallback for ROCm stability
from eligsir.utils.rocm_reloc_fallback import compute_relocation_cuda