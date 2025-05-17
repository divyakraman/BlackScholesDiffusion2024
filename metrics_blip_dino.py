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
from transformers import AutoProcessor, Blip2ForImageTextRetrieval
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
import requests
from transformers import AutoImageProcessor, Dinov2Model, CLIPProcessor, CLIPModel


device = "cuda" if torch.cuda.is_available() else "cpu"

model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16, cache_dir = './huggingface_models/')
processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g", cache_dir = './huggingface_models/')

model.to(device)

# DINO
image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", cache_dir = './huggingface_models/')
dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base", cache_dir = './huggingface_models/').cuda()


def blip_score(image, prompt):
    text = prompt
    inputs = processor(images=image, text=text, return_tensors="pt").to(device, torch.float16)
    itm_out = model(**inputs, use_image_text_matching_head=True)
    logits_per_image = torch.nn.functional.softmax(itm_out.logits_per_image, dim=1)
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    score1 = probs[0][1].detach().cpu().numpy()
    return score1

def dino_score(image_list1, image_list2):
    
    i = 0
    for image in image_list1:
        
        image1 = PIL.Image.open(image).convert("RGB")
        image_inputs1 = image_processor(image1, return_tensors="pt")
        image_inputs1['pixel_values'] = image_inputs1['pixel_values'].cuda()
        
        with torch.no_grad():
            outputs1 = dino_model(**image_inputs1)

        if(i==0):
            last_hidden_states1 = outputs1.last_hidden_state # 1, 257, 768
            i=1
        else:
            last_hidden_states1 = last_hidden_states1 + outputs1.last_hidden_state # 1, 257, 768
    
    i = 0
    for image in image_list2:
        
        image2 = PIL.Image.open(image).convert("RGB")
        image_inputs2 = image_processor(image2, return_tensors="pt")
        image_inputs2['pixel_values'] = image_inputs2['pixel_values'].cuda()
        
        with torch.no_grad():
            outputs2 = dino_model(**image_inputs2)

        if(i==0):
            last_hidden_states2 = outputs2.last_hidden_state # 1, 257, 768
            i=1
        else:
            last_hidden_states2 = last_hidden_states2 + outputs2.last_hidden_state # 1, 257, 768
    
    last_hidden_states1 = last_hidden_states1/len(image_list1)
    last_hidden_states2 = last_hidden_states2/len(image_list1)
        
    dino_score = F.cosine_similarity(last_hidden_states1.view(-1), last_hidden_states2.view(-1), dim=0)
    
    return dino_score


blip_total = 0
dino_total = 0
dino_blip = 0

with open('data/set1.txt', 'r') as f:
    prompts_list = f.readlines()


for i in range(len(prompts_list)):
    prompts = prompts_list[i]
    prompts = prompts.split('\t')
    file_name = prompts[0]
    prompts = prompts[1]
    prompts = prompts[1:-2]
    prompts = prompts.split(',')
    print(prompts)

    savedir_gen = './results/set1/' + file_name # Gen Save Dir
    savedir = './results/set1/' + file_name + '/bs/' # Black Scholes 
    
    image_list = glob.glob(savedir + '*.png')
    image_list_vanilla1 = glob.glob(savedir_gen + '/vanilla/text3/' + '*.png')
    image_list_vanilla2 = glob.glob(savedir_gen + '/vanilla/text4/' + '*.png')

    dino_vanilla1 = dino_score(image_list, image_list_vanilla1)
    dino_vanilla2 = dino_score(image_list, image_list_vanilla2)
    max_dino = 0.5 * (dino_vanilla1 + dino_vanilla2)
    dino_total = dino_total + max_dino


    max_blip = 0
    for image in image_list:
        image1 = PIL.Image.open(image).convert("RGB")

        blip3 = blip_score(image1, prompts[0])
        blip4 = blip_score(image1, prompts[1])
                 

        
        torch.cuda.empty_cache()

        max_blip = max_blip + 0.5 * (blip3 + blip4)

    max_blip = max_blip/5
    blip_total = blip_total + max_blip
    dino_blip = dino_blip + (max_dino * max_blip)
        
        
print("BLIP Total: ", blip_total/(len(prompts_list)))
print("DINO Total: ", dino_total/(len(prompts_list)))
print("DINO BLIP Score: ", dino_blip/(len(prompts_list)))

