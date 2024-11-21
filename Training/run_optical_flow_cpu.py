import pandas as pd
import torch
import torchvision
from train_utils.unimatch.unimatch.unimatch import UniMatch
import torch.nn.functional as F
from train_utils.dataset import WebVid10M
from tqdm.auto import tqdm
import os
from train_stage1 import get_optical_flows
from utils.flow_viz import save_vis_flow_tofile
import csv

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
    # cast flow to the highest precision
    flow = flow.to(torch.float64)
    # Get the center of the image
    center = (flow.shape[2] // 2, flow.shape[3] // 2)
    # Get the foreground region
    flow_foreground_abs = flow[
        :,
        :,
        center[0] - int(foreground_ratio * flow.shape[2] // 2):center[0] + int(foreground_ratio * flow.shape[2] // 2), 
        center[1] - int(foreground_ratio * flow.shape[3] // 2):center[1] + int(foreground_ratio * flow.shape[3] // 2)
    ].abs()
    
    flow_foreground_abs_mean = flow_foreground_abs.mean()
    
    flow_foreground_abs_sum = flow_foreground_abs.sum()
    flow_all_abs_sum = flow.abs().sum()
    
    flow_background_abs_sum = flow_all_abs_sum - flow_foreground_abs_sum
    flow_background_abs_mean = flow_background_abs_sum / (flow.numel() - flow_foreground_abs.numel())
    is_trackshot = flow_foreground_abs_mean < foreground_motion_threshold and flow_background_abs_mean > background_motion_threshold

    return is_trackshot, flow_foreground_abs_mean, flow_background_abs_mean

def main():
    # Force CPU
    device = torch.device('cpu')
    
    # Initialize model
    unimatch = UniMatch(feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task='flow').to(device)
    checkpoint = torch.load('./train_utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth', 
                          map_location=device)
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
        num_workers=4,
        shuffle=False
    )

    # Create output directory
    os.makedirs('optical_flow_outputs', exist_ok=True)
    
    # Create/open CSV file for writing results
    csv_path = os.path.join('optical_flow_outputs', 'trackshot_results.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['video_name', 'is_trackshot', 'avg_flow_foreground', 'avg_flow_background'])

    # Process videos
    for idx, batch in enumerate(tqdm(dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        video_name = batch["video_name"][0]  # Get video name from batch
        
        # Get optical flows
        flows = get_optical_flows(unimatch, pixel_values)

        # Save flows as .png visualizations
        flows = flows.squeeze() # Remove batch dim but keep as tensor

        is_trackshot_result, avg_flow_foreground, avg_flow_background = is_trackshot(
            flows, FOREGROUND_SIZE_RATIO, FOREGROUND_MOTION_THRESHOLD, BACKGROUND_MOTION_THRESHOLD
        )
        
        # Write results to CSV
        csv_writer.writerow([
            video_name,
            int(is_trackshot_result),  # Convert bool to int for CSV
            float(avg_flow_foreground),  # Convert tensor to float
            float(avg_flow_background)
        ])
    
    csv_file.close()
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()

# example command:
# python run_optical_flow_cpu.py