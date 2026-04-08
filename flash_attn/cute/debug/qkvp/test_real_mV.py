import cutlass
import cutlass.cute as cute
from cutlass.utils.blockscaled_layout import BlockScaledBasicChunk

@cute.jit
def test():
    chunk = BlockScaledBasicChunk(16).layout
    cute.printf("chunk shape={} stride={}", chunk.shape, chunk.stride)
    # REAL mV.shape after kernel's make_ordered_layout is (b, s, h, d)
    sh = (1, 8192, 16, 128)
    l = cute.tile_to_shape(chunk, sh, (2, 1, 3, 4))
    cute.printf("mV (b,s,h,d)=(1,8192,16,128) order=(2,1,3,4):")
    cute.printf("  shape={}", l.shape)
    cute.printf("  stride={}", l.stride)
    cute.printf("  cosize={}", cute.cosize(l))
test()
