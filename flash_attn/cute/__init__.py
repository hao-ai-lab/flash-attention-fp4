"""Flash Attention CUTE (CUDA Template Engine) implementation."""

__version__ = "0.1.0"

# Move editable finder to front of sys.meta_path to ensure local files are used
# This must happen before any imports
import sys

for i, finder in enumerate(sys.meta_path):
    finder_type = type(finder)
    finder_module = getattr(finder_type, '__module__', '')
    if 'flash_attn_cute' in finder_module and 'EditableFinder' in finder_module:
        if i > 0:
            sys.meta_path.insert(0, sys.meta_path.pop(i))
        break

import cutlass.cute as cute

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

from flash_attn.cute.cute_dsl_utils import cute_compile_patched

# Patch cute.compile to optionally dump SASS
cute.compile = cute_compile_patched


__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
