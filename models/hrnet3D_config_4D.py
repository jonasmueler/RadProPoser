from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


HRNET3D_Tiny_Feat32 = CN()
HRNET3D_Tiny_Feat32.LAYER1 = CN()
HRNET3D_Tiny_Feat32.LAYER1.INPLANES = 32 # 32 for doppler dimension
HRNET3D_Tiny_Feat32.LAYER1.BLOCK = "ResNetBlock"

HRNET3D_Tiny_Feat32.STAGE2 = CN()
HRNET3D_Tiny_Feat32.STAGE2.INPLANES = 32 
HRNET3D_Tiny_Feat32.STAGE2.NUM_MODULES = 1
HRNET3D_Tiny_Feat32.STAGE2.NUM_BRANCHES = 2
HRNET3D_Tiny_Feat32.STAGE2.NUM_BLOCKS = [1, 1]
HRNET3D_Tiny_Feat32.STAGE2.NUM_CHANNELS = [32, 64]
HRNET3D_Tiny_Feat32.STAGE2.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat32.STAGE2.FUSE_METHOD = "SUM"

HRNET3D_Tiny_Feat32.STAGE3 = CN()
HRNET3D_Tiny_Feat32.STAGE3.NUM_MODULES = 1
HRNET3D_Tiny_Feat32.STAGE3.NUM_BRANCHES = 3
HRNET3D_Tiny_Feat32.STAGE3.NUM_BLOCKS = [1, 1, 1]
HRNET3D_Tiny_Feat32.STAGE3.NUM_CHANNELS = [32, 64, 128]
HRNET3D_Tiny_Feat32.STAGE3.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat32.STAGE3.FUSE_METHOD = "SUM"


HRNET3D_Tiny_Feat16_zyx = CN()
HRNET3D_Tiny_Feat16_zyx.LAYER1 = CN()
HRNET3D_Tiny_Feat16_zyx.LAYER1.INPLANES = 1 
HRNET3D_Tiny_Feat16_zyx.LAYER1.BLOCK = "ResNetBlock"

HRNET3D_Tiny_Feat16_zyx.STAGE2 = CN()
HRNET3D_Tiny_Feat16_zyx.STAGE2.INPLANES = 16
HRNET3D_Tiny_Feat16_zyx.STAGE2.NUM_MODULES = 1
HRNET3D_Tiny_Feat16_zyx.STAGE2.NUM_BRANCHES = 2
HRNET3D_Tiny_Feat16_zyx.STAGE2.NUM_BLOCKS = [1, 1]
HRNET3D_Tiny_Feat16_zyx.STAGE2.NUM_CHANNELS = [16, 32]
HRNET3D_Tiny_Feat16_zyx.STAGE2.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat16_zyx.STAGE2.FUSE_METHOD = "SUM"

HRNET3D_Tiny_Feat16_zyx.STAGE3 = CN()
HRNET3D_Tiny_Feat16_zyx.STAGE3.NUM_MODULES = 1
HRNET3D_Tiny_Feat16_zyx.STAGE3.NUM_BRANCHES = 3
HRNET3D_Tiny_Feat16_zyx.STAGE3.NUM_BLOCKS = [1, 1, 1]
HRNET3D_Tiny_Feat16_zyx.STAGE3.NUM_CHANNELS = [16, 32, 64]
HRNET3D_Tiny_Feat16_zyx.STAGE3.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat16_zyx.STAGE3.FUSE_METHOD = "SUM"



HRNET3D_Tiny_Feat16_zyx_l4 = CN()
HRNET3D_Tiny_Feat16_zyx_l4.LAYER1 = CN()
HRNET3D_Tiny_Feat16_zyx_l4.LAYER1.INPLANES = 1 
HRNET3D_Tiny_Feat16_zyx_l4.LAYER1.BLOCK = "ResNetBlock"

HRNET3D_Tiny_Feat16_zyx_l4.STAGE2 = CN()
HRNET3D_Tiny_Feat16_zyx_l4.STAGE2.INPLANES = 16
HRNET3D_Tiny_Feat16_zyx_l4.STAGE2.NUM_MODULES = 1
HRNET3D_Tiny_Feat16_zyx_l4.STAGE2.NUM_BRANCHES = 2
HRNET3D_Tiny_Feat16_zyx_l4.STAGE2.NUM_BLOCKS = [1, 1]
HRNET3D_Tiny_Feat16_zyx_l4.STAGE2.NUM_CHANNELS = [16, 32]
HRNET3D_Tiny_Feat16_zyx_l4.STAGE2.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat16_zyx_l4.STAGE2.FUSE_METHOD = "SUM"

HRNET3D_Tiny_Feat16_zyx_l4.STAGE3 = CN()
HRNET3D_Tiny_Feat16_zyx_l4.STAGE3.NUM_MODULES = 1
HRNET3D_Tiny_Feat16_zyx_l4.STAGE3.NUM_BRANCHES = 3
HRNET3D_Tiny_Feat16_zyx_l4.STAGE3.NUM_BLOCKS = [1, 1, 1]
HRNET3D_Tiny_Feat16_zyx_l4.STAGE3.NUM_CHANNELS = [16, 32, 64]
HRNET3D_Tiny_Feat16_zyx_l4.STAGE3.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat16_zyx_l4.STAGE3.FUSE_METHOD = "SUM"

HRNET3D_Tiny_Feat16_zyx_l4.STAGE4 = CN()
HRNET3D_Tiny_Feat16_zyx_l4.STAGE4.NUM_MODULES = 1
HRNET3D_Tiny_Feat16_zyx_l4.STAGE4.NUM_BRANCHES =4
HRNET3D_Tiny_Feat16_zyx_l4.STAGE4.NUM_BLOCKS = [1, 1, 1, 1]
HRNET3D_Tiny_Feat16_zyx_l4.STAGE4.NUM_CHANNELS = [16, 32, 64, 64]
HRNET3D_Tiny_Feat16_zyx_l4.STAGE4.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat16_zyx_l4.STAGE4.FUSE_METHOD = "SUM"

HRNET3D_Tiny_Feat32_zyx_l4 = CN()
HRNET3D_Tiny_Feat32_zyx_l4.LAYER1 = CN()
HRNET3D_Tiny_Feat32_zyx_l4.LAYER1.INPLANES = 1 
HRNET3D_Tiny_Feat32_zyx_l4.LAYER1.BLOCK = "ResNetBlock"

HRNET3D_Tiny_Feat32_zyx_l4.STAGE2 = CN()
HRNET3D_Tiny_Feat32_zyx_l4.STAGE2.INPLANES = 32
HRNET3D_Tiny_Feat32_zyx_l4.STAGE2.NUM_MODULES = 1
HRNET3D_Tiny_Feat32_zyx_l4.STAGE2.NUM_BRANCHES = 2
HRNET3D_Tiny_Feat32_zyx_l4.STAGE2.NUM_BLOCKS = [1, 1]
HRNET3D_Tiny_Feat32_zyx_l4.STAGE2.NUM_CHANNELS = [32, 32]
HRNET3D_Tiny_Feat32_zyx_l4.STAGE2.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat32_zyx_l4.STAGE2.FUSE_METHOD = "SUM"

HRNET3D_Tiny_Feat32_zyx_l4.STAGE3 = CN()
HRNET3D_Tiny_Feat32_zyx_l4.STAGE3.NUM_MODULES = 1
HRNET3D_Tiny_Feat32_zyx_l4.STAGE3.NUM_BRANCHES = 3
HRNET3D_Tiny_Feat32_zyx_l4.STAGE3.NUM_BLOCKS = [1, 1, 1]
HRNET3D_Tiny_Feat32_zyx_l4.STAGE3.NUM_CHANNELS = [32, 32, 64]
HRNET3D_Tiny_Feat32_zyx_l4.STAGE3.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat32_zyx_l4.STAGE3.FUSE_METHOD = "SUM"

HRNET3D_Tiny_Feat32_zyx_l4.STAGE4 = CN()
HRNET3D_Tiny_Feat32_zyx_l4.STAGE4.NUM_MODULES = 1
HRNET3D_Tiny_Feat32_zyx_l4.STAGE4.NUM_BRANCHES =4
HRNET3D_Tiny_Feat32_zyx_l4.STAGE4.NUM_BLOCKS = [1, 1, 1, 1]
HRNET3D_Tiny_Feat32_zyx_l4.STAGE4.NUM_CHANNELS = [32, 32, 64, 64]
HRNET3D_Tiny_Feat32_zyx_l4.STAGE4.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat32_zyx_l4.STAGE4.FUSE_METHOD = "SUM"


HRNET3D_Tiny_Feat32_zyx_l4_IN32 = CN()
HRNET3D_Tiny_Feat32_zyx_l4_IN32.LAYER1 = CN()
HRNET3D_Tiny_Feat32_zyx_l4_IN32.LAYER1.INPLANES = 32
HRNET3D_Tiny_Feat32_zyx_l4_IN32.LAYER1.BLOCK = "ResNetBlock"

HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE2 = CN()
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE2.INPLANES = 32
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE2.NUM_MODULES = 1
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE2.NUM_BRANCHES = 2
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE2.NUM_BLOCKS = [1, 1]
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE2.NUM_CHANNELS = [32, 32]
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE2.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE2.FUSE_METHOD = "SUM"

HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE3 = CN()
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE3.NUM_MODULES = 1
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE3.NUM_BRANCHES = 3
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE3.NUM_BLOCKS = [1, 1, 1]
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE3.NUM_CHANNELS = [32, 32, 64]
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE3.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE3.FUSE_METHOD = "SUM"

HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE4 = CN()
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE4.NUM_MODULES = 1
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE4.NUM_BRANCHES =4
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE4.NUM_BLOCKS = [1, 1, 1, 1]
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE4.NUM_CHANNELS = [32, 32, 64, 64]
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE4.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat32_zyx_l4_IN32.STAGE4.FUSE_METHOD = "SUM"




HRNET3D_Tiny_Feat64_zyx_l4_IN64 = CN()
HRNET3D_Tiny_Feat64_zyx_l4_IN64.LAYER1 = CN()
HRNET3D_Tiny_Feat64_zyx_l4_IN64.LAYER1.INPLANES = 128 * 2
HRNET3D_Tiny_Feat64_zyx_l4_IN64.LAYER1.BLOCK = "ResNetBlock"

HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE2 = CN()
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE2.INPLANES = 64
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE2.NUM_MODULES = 1
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE2.NUM_BRANCHES = 2
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE2.NUM_BLOCKS = [1, 1]
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE2.NUM_CHANNELS = [64, 64]
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE2.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE2.FUSE_METHOD = "SUM"

HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE3 = CN()
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE3.NUM_MODULES = 1
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE3.NUM_BRANCHES = 3
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE3.NUM_BLOCKS = [1, 1, 1]
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE3.NUM_CHANNELS = [64, 64, 128]
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE3.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE3.FUSE_METHOD = "SUM"

HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE4 = CN()
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE4.NUM_MODULES = 1
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE4.NUM_BRANCHES =4
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE4.NUM_BLOCKS = [1, 1, 1, 1]
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE4.NUM_CHANNELS = [64, 64, 128, 128]
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE4.BLOCK = "ResNetBlock"
HRNET3D_Tiny_Feat64_zyx_l4_IN64.STAGE4.FUSE_METHOD = "SUM"


MODEL_CONFIGS = {
    "hr_tiny_feat32": HRNET3D_Tiny_Feat32,
    "hr_tiny_feat16_zyx": HRNET3D_Tiny_Feat16_zyx,
    "hr_tiny_feat16_zyx_l4": HRNET3D_Tiny_Feat16_zyx_l4,
    "hr_tiny_feat32_zyx_l4": HRNET3D_Tiny_Feat32_zyx_l4,
    "hr_tiny_feat32_zyx_l4_in32": HRNET3D_Tiny_Feat32_zyx_l4_IN32,
    "hr_tiny_feat64_zyx_l4_in64": HRNET3D_Tiny_Feat64_zyx_l4_IN64
}
