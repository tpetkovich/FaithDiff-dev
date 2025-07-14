from utils.system import quantize_8bit
from huggingface_hub import hf_hub_download
from .pipelines.pipeline_FaithDiff_tlc import FaithDiffStableDiffusionXLPipeline
import torch
from diffusers import AutoencoderKL
from .models.unet_2d_condition_vae_extension import UNet2DConditionModel
from diffusers import AutoencoderKL, DDPMScheduler
from .models.bsrnet_arch import RRDBNet as BSRNet


def FaithDiff_pipeline(sdxl_path, VAE_FP16_path, FaithDiff_path, use_fp8 = False):
    dtype = torch.float16
    vae = AutoencoderKL.from_pretrained(VAE_FP16_path).to(dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(sdxl_path, subfolder="unet", variant="fp16")

    model_file = hf_hub_download(FaithDiff_path, filename="FaithDiff.bin")
    print(model_file)
    unet.load_additional_layers(weight_path=model_file, dtype=dtype)    
    if use_fp8:
        quantize_8bit(unet)
    else:
        unet = unet.to(dtype=torch.float16)

    DDPM_scheduler = DDPMScheduler.from_pretrained(sdxl_path, subfolder="scheduler")
    pipe = FaithDiffStableDiffusionXLPipeline.from_pretrained(
        sdxl_path,
        vae = vae,
        add_sample = True,        
        denoise_encoder = unet.denoise_encoder,
        DDPM_scheduler = DDPM_scheduler,
        add_watermarker=False,
        torch_dtype=dtype,
        variant="fp16"
    )
    pipe.unet = unet

    return pipe

def create_bsrnet(bsrnet_path):
    bsrnet = BSRNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
    bsrnet.load_state_dict(torch.load(bsrnet_path), strict=True)
    return bsrnet