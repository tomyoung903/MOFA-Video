import torch
import torchvision
from train_utils.unimatch.unimatch.unimatch import UniMatch
import torch.nn.functional as F
from train_utils.dataset import WebVid10M
from tqdm.auto import tqdm
import os
from train_stage1 import get_optical_flows
from utils.flow_viz import save_vis_flow_tofile

FOREGROUND_SIZE_RATIO = 0.5 # Fraction of the image that is considered foreground in the center
FOREGROUND_MOTION_THRESHOLD = 3 # Threshold for the foreground motion (pixels per frame). If the average motion of the foreground is below this threshold, it is considered not moving.
BACKGROUND_MOTION_THRESHOLD = 2 # Threshold for the background motion (pixels per frame). If the average motion of the background is above this threshold, it is considered moving.

def is_trackshot(
    flow: torch.Tensor,  # size: (t, 2, h, w)
    foreground_ratio: float, 
    foreground_motion_threshold: float,
    background_motion_threshold: float
) -> bool:
    '''
    Check if the foreground has no motion by averaging the optical flow (absolute value)
    over the foreground region and checking if it is below the threshold.
    '''
    # Get the center of the image
    center = (flow.shape[2] // 2, flow.shape[3] // 2)
    # Get the foreground region
    avg_flow_foreground = flow[
        :,
        :,
        center[0] - int(foreground_ratio * flow.shape[2] // 2):center[0] + int(foreground_ratio * flow.shape[2] // 2), 
        center[1] - int(foreground_ratio * flow.shape[3] // 2):center[1] + int(foreground_ratio * flow.shape[3] // 2)
    ].abs().mean()
    avg_flow_all = flow.abs().mean()
    avg_flow_background = avg_flow_all - avg_flow_foreground
    
    return avg_flow_foreground < foreground_motion_threshold and avg_flow_background > background_motion_threshold


def main():
    # Initialize model
    unimatch = UniMatch(feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task='flow').to('cuda')
    checkpoint = torch.load('./train_utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
    unimatch.load_state_dict(checkpoint['model'])
    unimatch.eval()
    unimatch.requires_grad_(False)


    # Initialize dataset
    dataset = WebVid10M(
        meta_path='/home/tom/Open-Sora-dev/pose_traj_classes_2024-09-24_11-13-53_12_classes_intersect_pruned_5000_train.csv',
        sample_size=[720, 1280],
        sample_n_frames=6,
        sample_stride=4
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=4
    )

    # Create output directory
    os.makedirs('optical_flow_outputs', exist_ok=True)

    # Process videos
    for idx, batch in enumerate(tqdm(dataloader)):
        pixel_values = batch["pixel_values"].cuda()
        # Save video frames
        # for t in range(pixel_values.shape[1]):
        #     frame_t = pixel_values[0, t]  # Keep in [C,H,W] format for torchvision.utils.save_image
        #     torchvision.utils.save_image(frame_t, f'optical_flow_outputs/frame_{idx}_frame_{t}.png')
        
        # Get optical flows
        flows = get_optical_flows(unimatch, pixel_values)

        # Save flows as .png visualizations
        flows = flows.squeeze() # Remove batch dim but keep on GPU/as tensor
        # for t in range(flows.shape[0]):
        #     flow_t = flows[t].permute(1,2,0) # [H,W,2] format for vis_flow
        #     # Save flow visualization as PNG using flow_viz
        #     output_path = f'optical_flow_outputs/flow_{idx}_frame_{t}.png'
        #     save_vis_flow_tofile(flow_t, output_path)
        
        # # Save flows
        # torch.save(flows, f'optical_flow_outputs/flow_{idx}.pt')
        # break

        if is_trackshot(flows, FOREGROUND_SIZE_RATIO, FOREGROUND_MOTION_THRESHOLD, BACKGROUND_MOTION_THRESHOLD):
            print(f"{batch['video_name']}")

if __name__ == "__main__":
    main()
