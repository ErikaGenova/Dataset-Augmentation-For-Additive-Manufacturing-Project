import os
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import torch

# === CONFIG ===
input_root = "/content/mla_project/images/original"
output_root = "/content/mla_project/images/diffusionmodels"
categories = ["Defects", "NoDefects"]
images_per_input = 2
strength = 0.15
guidance_scale = 2.5

# === Setup pipeline ===
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# === Genera immagini per ciascuna categoria ===
for category in categories:
    input_dir = os.path.join(input_root, category)
    output_dir = os.path.join(output_root, category)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"Errore nel caricamento: {filename}")
            continue

        # Resize to nearest lower multiple of 64
        w, h = image.size
        w = (w // 64) * 64
        h = (h // 64) * 64
        image = image.resize((w, h))

        # Genera N immagini
        for i in range(images_per_input):
            result = pipe(prompt="", image=image, strength=strength, guidance_scale=guidance_scale).images[0]
            out_name = f"{os.path.splitext(filename)[0]}_aug{i+1}.png"
            out_path = os.path.join(output_dir, out_name)
            result.save(out_path)

        print(f"✔️ Generato da: {filename}")