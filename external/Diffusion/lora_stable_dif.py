from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os

# === CONFIG ===
input_root = "/content/mla_project/images/original"
output_root = "/content/mla_project/images/augmented/diffusion/StableDiffusion_finetuned"
lora_weights_path = "/content/mla_project/lora_trained_model/pytorch_lora_weights.safetensors"

prompt = "a grayscale high-resolution technical photo of a metallic powder bed surface with small circular defects, uneven texture, light and dark irregular spots, industrial process, machine vision"
negative_prompt = "colorful, artistic, cartoon"
num_images_per_input = 1
strength = 0.25
guidance_scale = 5.0
num_inference_steps = 40

# === Carica pipeline ===
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights(lora_weights_path)

# === Scorri categorie ===
categories = ["Defects", "NoDefects"]
for category in categories:
    input_dir = os.path.join(input_root, category)
    output_dir = os.path.join(output_root, category)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        input_path = os.path.join(input_dir, filename)
        try:
            image = Image.open(input_path).convert("RGB")
        except:
            print(f"❌ Errore su {input_path}")
            continue

        # Ridimensiona
        w, h = image.size
        image = image.resize(((w // 64) * 64, (h // 64) * 64))

        # Genera immagini
        for i in range(num_images_per_input):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images[0]

            out_name = f"{os.path.splitext(filename)[0]}_aug{i+1}.png"
            out_path = os.path.join(output_dir, out_name)
            result.save(out_path)

        print(f"✅ Generato da: {filename} → {num_images_per_input} immagini")
