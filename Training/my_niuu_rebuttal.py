
import cv2

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger

from train_utils.unimatch.unimatch.unimatch import UniMatch
from train_utils.unimatch.utils.flow_viz import flow_to_image

from models.softsplat import softsplat


logger = get_logger(__name__, log_level="INFO")


def preprocess_size(image1, image2, padding_factor=32):
    '''
        img: [b, c, h, w]
    '''
    transpose_img = False
    # the model is trained with size: width > height
    if image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True

    # inference_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
    #                 int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        
    inference_size = [384, 512]

    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)
    
    return image1, image2, inference_size, ori_size, transpose_img


def postprocess_size(flow_pr, inference_size, ori_size, transpose_img):

    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

    if transpose_img:
        flow_pr = torch.transpose(flow_pr, -2, -1)
    
    return flow_pr


@torch.no_grad()
def get_optical_flows(unimatch, image1, image2):
    '''
        video_frame: [b, t, c, w, h]
    '''

    # print(video_frame.dtype)

    # print(image1.dtype)
    image1_r, image2_r, inference_size, ori_size, transpose_img = preprocess_size(image1, image2)
    # print(image1_r.dtype)
    results_dict_r = unimatch(image1_r, image2_r,
        attn_type='swin',
        attn_splits_list=[2, 8],
        corr_radius_list=[-1, 4],
        prop_radius_list=[-1, 1],
        num_reg_refine=6,
        task='flow',
        pred_bidir_flow=False,
        )
    flow_r = results_dict_r['flow_preds'][-1]  # [b, 2, H, W]
    # print(flow_r.shape)
    flow = postprocess_size(flow_r, inference_size, ori_size, transpose_img)
    
    return flow




def softmax_splatting_from_unimatch(unimatch, path0, path1):

    frame0 = torch.from_numpy(np.array(Image.open(path0).convert('RGB'))).cuda().permute(2, 0, 1).unsqueeze(0).float()
    frame1 = torch.from_numpy(np.array(Image.open(path1).convert('RGB'))).cuda().permute(2, 0, 1).unsqueeze(0).float()

    flows = get_optical_flows(unimatch, frame0, frame1)[0]

    flow_img = flow_to_image(flows.permute(1, 2, 0))

    tenOne, tenFlow = frame0 / 255, flows[None, :]

    return tenOne, tenFlow, flow_img


if __name__ == '__main__':
    # Define Unimatch for optical flow prediction
    unimatch = UniMatch(feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task='flow').to('cuda')
    
    checkpoint = torch.load('./Training/train_utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
    unimatch.load_state_dict(checkpoint['model'])
    unimatch.eval()
    unimatch.requires_grad_(False)
    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance


    '''for image'''
    path0 = "Training/one.png"
    path1 = "Training/two.png"
    softmax_flow_path = "Training/flow.flo"
    tenOne0, tenFlow0, unimatch_flow_img = softmax_splatting_from_unimatch(unimatch, path0, path1)
    
    
    Image.fromarray(unimatch_flow_img).save('unimatch_flow.png')
    tenAverage0 = softsplat(tenIn=tenOne0, tenFlow=tenFlow0, tenMetric=None, strMode='avg')
    
    result0 = np.uint8(tenAverage0[0, :, :, :].cpu().numpy().transpose(1, 2, 0)*255)

    all_together = np.concatenate([
        np.array(Image.open(path0).convert('RGB')), 
        np.array(Image.open(path1).convert('RGB')), 
        unimatch_flow_img, result0
    ], axis=1)

    Image.fromarray(all_together).save('average.png')
    
    '''for video'''
    
