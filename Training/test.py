import torch
from PIL import Image
import numpy as np

path0 = '/home/tom/Open-Sora-dev/tests/unit/motion_brush/frames/frame_00000.jpg'

frame0 = torch.from_numpy(np.array(Image.open(path0).convert('RGB'))).cuda().permute(2, 0, 1).unsqueeze(0).float()

print(frame0.shape)
