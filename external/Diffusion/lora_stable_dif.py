from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
import torch
import os

# === Percorsi e setup ===
lora_weights_path = "/content/mla_project/lora_trained_model/pytorch_lora_weights.safetensors"
output_dir = "/content/mla_project/images/diffusionmodels/lora-stable-dif"
os.makedirs(f"{output_dir}/Defects", exist_ok=True)
os.makedirs(f"{output_dir}/NoDefects", exist_ok=True)

# === Carica pipeline base ===
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# === Carica e applica LoRA ===
pipe.load_lora_weights(load_file(lora_weights_path))
pipe.fuse_lora()

# === Prompt ===
prompts = {
    "Defects": "a grayscale industrial powder bed surface with subtle defects, metallic texture, high detail, technical photo",
    "NoDefects": "a grayscale industrial powder bed surface without defects, smooth metallic texture, high detail, technical photo"
}

# === Genera immagini ===
for label, prompt in prompts.items():
    for i in range(5):
        image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=8).images[0]
        image.save(f"{output_dir}/{label}/image_{label.lower()}_{i+1}.png")
        print(f"✔️ Salvata: {output_dir}/{label}/image_{label.lower()}_{i+1}.png")

print("✅ Generazione completata.")
