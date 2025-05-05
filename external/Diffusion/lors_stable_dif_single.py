import argparse
import os
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import torch

def generate_augmented_images(image_path: str, num_generations: int, output_root: str):
    lora_weights_path = "/content/mla_project/lora_trained_model/pytorch_lora_weights.safetensors"

    prompt = "a grayscale high-resolution technical photo of a metallic powder bed surface with small circular defects, uneven texture, light and dark irregular spots, industrial process, machine vision"
    negative_prompt = "colorful, artistic, cartoon"
    strength = 0.15
    guidance_scale = 5.0
    num_inference_steps = 40

    # === Carica pipeline ===
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to("cuda")
    pipe.load_lora_weights(lora_weights_path)

    # === Carica immagine ===
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    image = image.resize(((w // 64) * 64, (h // 64) * 64))

    # === Costruisci path output ===
    category = "Defects" if "Defects" in image_path else "NoDefects"
    output_dir = os.path.join(output_root, category)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # === Genera immagini ===
    for i in range(num_generations):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]

        out_name = f"{base_name}_aug{i+1}.png"
        result.save(os.path.join(output_dir, out_name))

    print(f"✅ {num_generations} immagini generate da {image_path} → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Percorso dell'immagine da usare come base")
    parser.add_argument("--num_generations", type=int, default=1, help="Numero di immagini da generare")
    parser.add_argument("--output_dir", required=True, help="Cartella di destinazione per le immagini generate")

    args = parser.parse_args()
    generate_augmented_images(args.image_path, args.num_generations, args.output_dir)
