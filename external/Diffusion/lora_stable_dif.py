from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.loaders import AttnProcsLayers
import torch
from PIL import Image
import os

# === PERCORSI ===
img_path = "/content/mla_project/images/original/Defects/Image0.jpg"
lora_weights_path = "/content/mla_project/lora_trained_model/pytorch_lora_weights.safetensors"
output_dir = "/content/mla_project/images/diffusionmodels/LoRA_img2img"
os.makedirs(output_dir, exist_ok=True)

# === PROMPT E PARAMETRI ===
prompt = "a grayscale high-resolution technical photo of a metallic powder bed surface with small circular defects, uneven texture, light and dark irregular spots, industrial process, machine vision"
negative_prompt = "colorful, artistic, cartoon"
num_images = 5
strength = 0.25
guidance_scale = 5.0
num_inference_steps = 40

# === CARICA LA PIPELINE BASE ===
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# === CARICA PESI LoRA ===
pipe.load_lora_weights(lora_weights_path)

# === CARICA E RIDIMENSIONA IMMAGINE ===
image = Image.open(img_path).convert("RGB")
w, h = image.size
image = image.resize(((w // 64) * 64, (h // 64) * 64))

# === GENERA IMMAGINI ===
for i in range(num_images):
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]

    result.save(os.path.join(output_dir, f"img_aug_{i+1}.png"))
