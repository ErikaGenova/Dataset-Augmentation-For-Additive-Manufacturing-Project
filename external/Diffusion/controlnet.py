import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector

# === CONFIG ===
input_root = "/content/mla_project/images/original"
output_root = "/content/mla_project/images/diffusion/ControlNet"
categories = {
    "Defects": 3,
    "NoDefects": 2
}
guidance_scale = 3.5
strength = 0.3
num_inference_steps = 30

prompt_dict = {
    "Defects": "top-down grayscale photograph of a metal powder bed used in additive manufacturing, with dark circular defects and irregular surface marks, realistic technical style",
    "NoDefects": "top-down grayscale photograph of a smooth, undisturbed metal powder bed surface, flat and uniform, without marks or defects, realistic technical documentation style"
}

# === Carica pipeline una sola volta
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
).to("cuda")

# === Detector per edge
canny = CannyDetector()

# === Generazione
for category, max_images in categories.items():
    input_dir = os.path.join(input_root, category)
    output_dir = os.path.join(output_root, category)
    os.makedirs(output_dir, exist_ok=True)

    images_done = 0
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, filename)
        try:
            init_image = Image.open(img_path).convert("RGB")
        except:
            print(f"❌ Errore nel caricamento: {filename}")
            continue

        w, h = init_image.size
        init_image = init_image.resize(((w // 64) * 64, (h // 64) * 64))

        control_image = canny(init_image)

        result = pipe(
            prompt=prompt_dict[category],
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength
        ).images[0]

        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, f"{base_name}_aug1.png")
        canny_path = os.path.join(output_dir, f"{base_name}_canny.png")

        result.save(out_path)
        control_image.save(canny_path)

        print(f"✅ Generato: {out_path}")
        images_done += 1
        if images_done >= max_images:
            break
