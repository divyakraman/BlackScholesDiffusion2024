import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import requests
from PIL import Image
from io import BytesIO
import torch
from diffusers import DiffusionPipeline, DDIMScheduler
import PIL
import cv2
import numpy as np 
from scipy import ndimage 
#import matplotlib.pyplot as plt 

has_cuda = torch.cuda.is_available()

device = torch.device('cpu' if not has_cuda else 'cuda')
torch.hub.set_dir('/scratch0/')

with open('data/set4.txt', 'r') as f:
    prompts_list = f.readlines()

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
        safety_checker=None,
    use_auth_token=False,
    custom_pipeline='./models/vanilla', cache_dir = 'dir_name',
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
).to(device)

generator = torch.Generator("cuda").manual_seed(0)
seed = 0

for i in range(len(prompts_list)):
    prompts = prompts_list[i]
    prompts = prompts.split('\t')
    file_name = prompts[0]
    prompts = prompts[1]
    prompts = prompts[1:-2]
    prompts = prompts.split(',')
    print(prompts)
    
    savedir = './results/set4/' + file_name + '/vanilla/text1/'

    os.makedirs(savedir, exist_ok=True)
    eval_prompt = prompts[0]
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result1.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result2.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result3.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result4.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result5.png')

    savedir = './results/set4/' + file_name + '/vanilla/text2/'
    os.makedirs(savedir, exist_ok=True)
    eval_prompt = prompts[1]
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result1.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result2.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result3.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result4.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result5.png')

    '''
    savedir = './results/set1/' + file_name + '/vanilla/text3/'
    os.makedirs(savedir, exist_ok=True)
    eval_prompt = prompts[2]
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result1.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result2.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result3.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result4.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result5.png')

    savedir = './results/set1/' + file_name + '/vanilla/text4/'
    os.makedirs(savedir, exist_ok=True)
    eval_prompt = prompts[3]
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result1.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result2.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result3.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result4.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result5.png')
    '''
