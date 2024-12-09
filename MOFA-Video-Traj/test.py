import os
os.chdir("MOFA-Video-Traj")
from PIL import Image
import numpy as np
from run_gradio import Drag
import torchvision.transforms as transforms
from utils.utils import image2pil

drag = Drag("cuda:0", 512, 512, 25)

drag.run(
    "/home/tom/MOFA-Video/MOFA-Video-Traj/test.png",
    [
        ((197, 198), (239, 182)), # the first trajectory starts from (197, 198) and ends at (239, 182)
        ((459, 139), (427, 101)) # the second trajectory starts from (459, 139) and ends at (427, 101)
    ],
    1,
    motion_brush_mask=np.zeros((512, 512)),
    motion_brush_viz=np.zeros((512, 512, 4)),
    ctrl_scale=0.6
)

