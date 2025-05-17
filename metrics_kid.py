#https://huggingface.co/docs/transformers/main/en/model_doc/dinov2
#https://github.com/facebookresearch/sscd-copy-detection
#https://huggingface.co/docs/diffusers/conceptual/evaluation

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
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from torchmetrics.image.kid import KernelInceptionDistance

kid = KernelInceptionDistance(subset_size=5)

def kid_score(image_list1, image_list2):
    
    for image in image_list1:
        
        image1 = PIL.Image.open(image).convert("RGB")
        image1 = np.asarray(image1)
        image1 = torch.from_numpy(image1)
        image1 = image1.unsqueeze(0)
        image1 = torch.permute(image1, (0,3,1,2))        
        kid.update(image1, real=False)
    
    for image in image_list2:
        
        image1 = PIL.Image.open(image).convert("RGB")
        image1 = np.asarray(image1)
        image1 = torch.from_numpy(image1)
        image1 = image1.unsqueeze(0)
        image1 = torch.permute(image1, (0,3,1,2))        
        kid.update(image1, real=True)
    
    
    kid_score = kid.compute()
    kid_score = kid_score[0]
    kid_score = kid_score.detach().cpu().numpy()

    return kid_score


kid_total = 0

with open('data/set4.txt', 'r') as f:
    prompts_list = f.readlines()

for i in range(len(prompts_list)):
    prompts = prompts_list[i]
    prompts = prompts.split('\t')
    file_name = prompts[0]
    prompts = prompts[1]
    prompts = prompts[1:-2]
    prompts = prompts.split(',')
    print(prompts)

    savedir_gen = './results/set4/' + file_name # Gen Save Dir
    savedir = './results/set4/' + file_name + '/bs/' # Black Scholes 

    
    image_list = glob.glob(savedir + '*.png')
    image_list_vanilla1 = glob.glob(savedir_gen + '/vanilla/text3/' + '*.png')
    image_list_vanilla2 = glob.glob(savedir_gen + '/vanilla/text4/' + '*.png')

    kid_vanilla1 = kid_score(image_list, image_list_vanilla1)
    kid_vanilla2 = kid_score(image_list, image_list_vanilla2)

    kid_total = kid_total + 0.5 * (kid_vanilla1 + kid_vanilla2)
    
        
print("KID Total: ", kid_total/(len(prompts_list)))
