import os
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import torch

# === CONFIG ===
input_root = "/content/mla_project/images/original"
output_root = "/content/mla_project/images/diffusion/StableDiffusionTest"
categories = {
    "Defects": 3,
    "NoDefects": 2
}
images_per_input = 1
strength = 0.15
guidance_scale = 2.5

# === Setup pipeline ===
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# === Genera immagini ===
for category, max_images in categories.items():
    input_dir = os.path.join(input_root, category)
    output_dir = os.path.join(output_root, category)
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        if count >= max_images:
            break

        img_path = os.path.join(input_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"❌ Errore nel caricamento: {filename}")
            continue

        # Ridimensiona ai multipli inferiori di 64
        w, h = image.size
        image = image.resize(((w // 64) * 64, (h // 64) * 64))

        for i in range(images_per_input):
            result = pipe(prompt="A top-down grayscale photograph of a metal powder bed used in additive manufacturing, featuring a flat rectangular build area with fine texture. The surface is partially disturbed by defects, including subtle dark circular marks and irregular patterns. Industrial lighting reflections appear at the edges. The background includes mechanical frame elements and a powder spreader at the top. Realistic, technical style.",
 image=image, strength=strength, guidance_scale=guidance_scale).images[0]
            out_name = f"{os.path.splitext(filename)[0]}_aug{i+1}.png"
            out_path = os.path.join(output_dir, out_name)
            result.save(out_path)

        print(f"✔️ Generato da: {filename}")
        count += 1
