"""
Wrapper to make flash_attn RotaryEmbedding compatible across versions
and add a safe wrap_triton shim for PyTorch < 2.5.
"""
import inspect
import torch

# PyTorch < 2.5 shim
if not hasattr(torch.library, 'wrap_triton'):
    def wrap_triton(kernel):
        return kernel
    torch.library.wrap_triton = wrap_triton
    print("Applied torch.library.wrap_triton compatibility patch for PyTorch < 2.5")

from flash_attn.layers.rotary import RotaryEmbedding as _BaseRotary

class CompatibleRotaryEmbedding(_BaseRotary):
    def __init__(self, *args, pos_idx_in_fp32=True, **kwargs):
        sig = inspect.signature(_BaseRotary.__init__)
        if 'pos_idx_in_fp32' in sig.parameters:
            # Pass through so the base class actually uses it
            super().__init__(*args, pos_idx_in_fp32=pos_idx_in_fp32, **kwargs)
        else:
            # Older versions: call without it; keep an attribute for downstream code if needed
            super().__init__(*args, **kwargs)
            self.pos_idx_in_fp32 = pos_idx_in_fp32

# Monkey-patch the module symbol *after* definition
import flash_attn.layers.rotary
flash_attn.layers.rotary.RotaryEmbedding = CompatibleRotaryEmbedding
