# . Download the pretrained checkpoint folder of [SVD_xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) from huggingface to `./ckpts`.

import huggingface_hub
from huggingface_hub import snapshot_download

# Download the pretrained checkpoint to ./ckpts directory
snapshot_download(repo_id="MyNiuuu/MOFA-Video-Traj", local_dir="./ckpts")


# 2. Download the Unimatch checkpoint from [here](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth) and put it into `./train_utils/unimatch/pretrained`.

