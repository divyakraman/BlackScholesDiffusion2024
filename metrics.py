import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import glob
import torch 
from PIL import Image
_ = torch.manual_seed(42)
import PIL
import numpy as np
from torchmetrics.multimodal import CLIPScore
from transformers import AutoImageProcessor, Dinov2Model, CLIPProcessor, CLIPModel
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

from torchmetrics.image import StructuralSimilarityIndexMeasure

# Load models

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir = './huggingface_models/').cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir = './huggingface_models/')
#clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir = './huggingface_models/').cuda()
#clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir = './huggingface_models/')


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)


with open('data/set3.txt', 'r') as f:
    prompts_list = f.readlines()

prompts_list = prompts_list[:]
clip_score_total1 = 0 # With composition prompt
clip_score_total2 = 0 # With individual prompt, addition

for i in range(len(prompts_list)):
    prompts = prompts_list[i]
    prompts = prompts.split('\t')
    file_name = prompts[0]
    prompts = prompts[1]
    prompts = prompts[1:-2]
    prompts = prompts.split(',')
    print(prompts)

    #savedir = './results/set4/' + file_name + '/vanilla/text2/' # Vanilla stable diffusion 
    #savedir = './results/set4/' + file_name + '/bs/' # Black Scholes 
    #savedir = './results/set4/' + file_name + '/alternating_sampling/' # Alternating Sampling 
    savedir = './results/set3/' + file_name + '/lininterp/' # Linear Interpolation 
    #savedir = './results/set4/' + file_name + '/clip_min/' # Min CLIP Score
    #savedir = './results/set4/' + file_name + '/promptmixing_iccv/' # Prompt Mixing ICCV Paper 

    max_clip_score2 = 0 
    
    images_list = glob.glob(savedir + '*.png')
    for image in images_list:
        image1 = PIL.Image.open(image).convert("RGB")
        clip_inputs = clip_processor(text=[prompts[0]], images=image1, return_tensors="pt", padding=True)
        clip_inputs['pixel_values'] = clip_inputs['pixel_values'].cuda()
        clip_inputs['input_ids'] = clip_inputs['input_ids'].cuda()
        clip_inputs['attention_mask'] = clip_inputs['attention_mask'].cuda()
        outputs = clip_model(**clip_inputs)
        clip_score_composite1 = outputs.logits_per_image.abs().cpu().detach().numpy()
        #clip_score_total1 = clip_score_total1 + 0.01 * clip_score
        #print("CLIP Score is: ", logits_per_image)

        clip_inputs = clip_processor(text=[prompts[1]], images=image1, return_tensors="pt", padding=True)
        clip_inputs['pixel_values'] = clip_inputs['pixel_values'].cuda()
        clip_inputs['input_ids'] = clip_inputs['input_ids'].cuda()
        clip_inputs['attention_mask'] = clip_inputs['attention_mask'].cuda()
        outputs = clip_model(**clip_inputs)
        clip_score_composite2 = outputs.logits_per_image.abs().cpu().detach().numpy()

        clip_inputs = clip_processor(text=[prompts[2]], images=image1, return_tensors="pt", padding=True)
        clip_inputs['pixel_values'] = clip_inputs['pixel_values'].cuda()
        clip_inputs['input_ids'] = clip_inputs['input_ids'].cuda()
        clip_inputs['attention_mask'] = clip_inputs['attention_mask'].cuda()
        outputs = clip_model(**clip_inputs)
        clip_score1 = outputs.logits_per_image.abs().cpu().detach().numpy()

        clip_inputs = clip_processor(text=[prompts[3]], images=image1, return_tensors="pt", padding=True)
        clip_inputs['pixel_values'] = clip_inputs['pixel_values'].cuda()
        clip_inputs['input_ids'] = clip_inputs['input_ids'].cuda()
        clip_inputs['attention_mask'] = clip_inputs['attention_mask'].cuda()
        outputs = clip_model(**clip_inputs)
        clip_score2 = outputs.logits_per_image.abs().cpu().detach().numpy()

        #print(clip_score1, clip_score2)

        temp_clip_score2 = 0.5 * (0.01 * clip_score1 + 0.01 * clip_score2)
        temp_clip_score1 = 0.005 * (clip_score_composite1 + clip_score_composite2)
        #if(temp_clip_score2 > max_clip_score2):
        #    max_clip_score2 = temp_clip_score2
        #    temp_clip_score1 = 0.005 * (clip_score_composite1 + clip_score_composite2)

        
        
        clip_score_total2 = clip_score_total2 + temp_clip_score2
        clip_score_total1 = clip_score_total1 + temp_clip_score1

        


print("CLIP Score, with composition prompt, is: ", clip_score_total1/(len(prompts_list)*5))
print("CLIP Score, with individual prompts, addition, is: ", clip_score_total2/(len(prompts_list)*5))
