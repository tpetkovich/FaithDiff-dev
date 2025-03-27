import gradio as gr
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.utils.checkpoint
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



parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, default='6688')
parser.add_argument("--no_llava", action='store_true', default=False)
args = parser.parse_args()

server_ip = args.ip
server_port = args.port
use_llava = not args.no_llava

if torch.cuda.device_count() >= 2:
    LLaVA_device = 'cuda:1'
    Diffusion_device = 'cuda:0'
elif torch.cuda.device_count() == 1:
    Diffusion_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')


pipe = FaithDiff_pipeline(sdxl_path=SDXL_PATH, VAE_FP16_path=VAE_FP16_PATH, FaithDiff_path=FAITHDIFF_PATH)
pipe = pipe.to(Diffusion_device)

### enable_vae_tiling
pipe.denoise_encoder.tile_sample_min_size = 1024
pipe.denoise_encoder.tile_overlap_factor = 0.25
pipe.denoise_encoder.enable_tiling()
pipe.vae.config.sample_size = 1024
pipe.vae.tile_overlap_factor = 0.25
pipe.vae.enable_tiling()

# load LLaVA
if use_llava:
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=LLaVA_device, load_8bit=True, load_4bit=False)
else:
    llava_agent = None


@torch.no_grad()
def caption_process(
    image: Image.Image,
    ) -> List[np.ndarray]:

    if use_llava:
        caption = llava_agent.gen_image_caption([image])
    else:
        caption = ['Caption Generation is not available. Please add text manually.']
    return caption[0]

@torch.no_grad()
def process(
    image: Image.Image,
    user_prompt: str,
    num_inference_steps: int,
    scale_factor: int,
    guidance_scale: float,
    seed: int,
    latent_tiled_size: int,
    latent_tiled_overlap: int,
    color_fix: str,
    start_point: str, 
    ) -> List[np.ndarray]:

    w, h = image.size
    w *= scale_factor
    h *= scale_factor
    image = image.resize((w, h), Image.LANCZOS)
    input_image, width_init, height_init, width_now, height_now = check_image_size(image)
    if use_llava:
        init_text = user_prompt
        words = init_text.split()
        words = words[3:]
        words[0] = words[0].capitalize()
        text = ' '.join(words)
        text = text.split('. ')
        text = '. '.join(text[:2]) + '.'

    user_prompt = text 
    negative_prompt_init = ""
    generator = torch.Generator(device=Diffusion_device).manual_seed(seed)
    gen_image = pipe(lr_img=input_image, prompt = user_prompt, negative_prompt = negative_prompt_init, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator, start_point=start_point, height = height_now, width=width_now,  overlap=latent_tiled_overlap, target_size=(latent_tiled_size, latent_tiled_size)).images[0]
    cropped_image = gen_image.crop((0, 0, width_init, height_init))
    if color_fix == 'nofix':
        out_image = cropped_image
    else:
        if color_fix == 'wavelet':
            out_image = wavelet_color_fix(cropped_image, image)
        elif color_fix == 'adain':
            out_image = adain_color_fix(cropped_image, image)


    images = []
    images.append(np.array(out_image))
    return images


#
MARKDOWN = \
"""
## FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-resolution

[GitHub](https://github.com/JyChen9811/FaithDiff/) | [Paper](https://arxiv.org/abs/2411.18824)

If FaithDiff is helpful for you, please help star the GitHub Repo. Thanks!
"""

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil")
            run_button = gr.Button(value="Restoration Run")
            llave_button = gr.Button(value="Caption Generation Run")
            with gr.Accordion("Options", open=True):
                user_prompt = gr.Textbox(label="User Prompt", value="")
                cfg_scale = gr.Slider(label="Classifier Free Guidance Scale (Set a value larger than 1 to enable it!)", minimum=0.1, maximum=10.0, value=5.5, step=0.1)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=50, value=20, step=1)
                seed = gr.Slider(label="Seed", minimum=0, maximum=88, step=1, value=42)
                latent_tiled_size = gr.Slider(label="Diffusion Tile Size", minimum=1024, maximum=1280, value=1024, step=1)
                latent_tiled_overlap = gr.Slider(label="Diffusion Tile Overlap", minimum=0.1, maximum=0.9, value=0.5, step=0.1)
                scale_factor = gr.Number(label="SR Scale", value=4)
                color_fix = gr.Dropdown(
                    label="Color Fix",
                    choices=["wavelet", "adain", "nofix"],
                    value="adain",  # default value
                    )
                start_point = gr.Dropdown(
                    label="Start Point",
                    choices=["lr", "noise"],
                    value="lr",  # default value
                    )
        with gr.Column():
            result_gallery = gr.Gallery(
                label="Output", show_label=False, columns=2, format="png"
            )

    inputs = [
        input_image,
        user_prompt,
        num_inference_steps,
        scale_factor,
        cfg_scale,
        seed,
        latent_tiled_size,
        latent_tiled_overlap,
        color_fix,
        start_point
    ]
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])
    llave_button.click(fn=caption_process, inputs=[input_image], outputs=[user_prompt])
block.launch(server_name=server_ip, server_port=server_port)
