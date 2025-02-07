import torch.cuda
import argparse
from FaithDiff.create_FaithDiff_model import FaithDiff_pipeline
from PIL import Image
from CKPT_PTH import LLAVA_MODEL_PATH, SDXL_PATH, FAITHDIFF_PATH, VAE_FP16_PATH
from utils.color_fix import wavelet_color_fix, adain_color_fix
from utils.image_process import check_image_size
from llava.llm_agent import LLavaAgent

import os
import numpy as np
import cv2




if torch.cuda.device_count() >= 2:
    LLaVA_device = 'cuda:1'
    Diffusion_device = 'cuda:0'
elif torch.cuda.device_count() == 1:
    Diffusion_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')


# hyparams here
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--upscale", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--min_size", type=int, default=1024)
parser.add_argument("--no_llava", action='store_true', default=False)
parser.add_argument("--use_tile_vae", action='store_true', default=False)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
parser.add_argument("--color_fix", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr')
args = parser.parse_args()
print(args)
use_llava = not args.no_llava

# load FaithDiff FP16
pipe = FaithDiff_pipeline(sdxl_path=SDXL_PATH, VAE_FP16_path=VAE_FP16_PATH, FaithDiff_path=FAITHDIFF_PATH)
pipe = pipe.to(Diffusion_device)
if args.use_tile_vae:
    pipe.vae.enable_tiling()

# load LLaVA
if use_llava:
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=LLaVA_device, load_8bit=args.load_8bit_llava, load_4bit=False)
else:
    llava_agent = None


os.makedirs(args.save_dir, exist_ok=True)

exist_file = os.listdir(args.save_dir)
with torch.no_grad():
    for file_name in sorted(os.listdir(args.img_dir)):
        img_name = file_name.split('.')[0]
        if f"{img_name}.png" in exist_file:
            print(f"{img_name}.png exist")
            continue
        else:
            print(img_name)
        image = Image.open(os.path.join(args.img_dir,file_name)).convert('RGB')


        # step 1: LLaVA
        if use_llava:
            captions = llava_agent.gen_image_caption([image])
        else:
            captions = ['']

        init_text = captions[0]
        words = init_text.split()
        words = words[3:]
        words[0] = words[0].capitalize()
        text = ' '.join(words)
        text = text.split('. ')
        text = '. '.join(text[:2]) + '.'
        print(text)

        # step 2: Restoration
        w, h = image.size
        w *= args.upscale
        h *= args.upscale
        image = image.resize((w, h), Image.BICUBIC)
        input_image, width_init, height_init, width_now, height_now = check_image_size(image)
        prompt_init = text 
        negative_prompt_init = ""
        generator = torch.Generator(device='cuda').manual_seed(args.seed)
        gen_image = pipe(lr_img=input_image, prompt = prompt_init, negative_prompt = negative_prompt_init, num_images_per_prompt=1, num_inference_steps=20, guidance_scale=5, generator=generator, start_point=args.start_point, schedule_path=SDXL_PATH, height = height_now, width=width_now, overlap=0.5, target_size=(1024, 1024)).images[0]
        path = os.path.join(args.save_dir, img_name+'.png')
        cropped_image = gen_image.crop((0, 0, width_init, height_init))
        if args.color_fix == 'nofix':
            out_image = cropped_image
        else:
            if args.color_fix == 'wavelet':
                out_image = wavelet_color_fix(cropped_image, image)
            elif args.color_fix == 'adain':
                out_image = adain_color_fix(cropped_image, image)
        out_image.save(path)




