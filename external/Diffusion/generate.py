from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os
import argparse


def generate_single_image(pipe, image_path, output_root, prompt, negative_prompt, num_images_per_input, strength, guidance_scale, num_inference_steps):
    """
    Generates a single image using the Stable Diffusion pipeline.

    Args:
        pipe: The Stable Diffusion pipeline object.
        image_path (str): Path to the input image.
        prompt (str): Prompt for image generation.
        negative_prompt (str): Negative prompt for image generation.
        strength (float): Strength of the image alteration.
        guidance_scale (float): Guidance scale for generation.
        num_inference_steps (int): Number of inference steps.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return

    # Resize the image to be divisible by 64
    w, h = image.size
    image = image.resize(((w // 64) * 64, (h // 64) * 64))

    # Construct output directory
    category = "Defects" if "Defects" in image_path else "NoDefects"
    output_dir = output_root
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Generate images
    for i in range(num_images_per_input):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]

        label = "0" if category == "NoDefects" else "1"
        out_name = f"{base_name}_{i}_class_{label}.png"

        result.save(os.path.join(output_dir, out_name))

    print(f"{num_images_per_input} images generated from {image_path} → {output_dir}")


def generate_images(pipe, input_root, output_root, prompt, negative_prompt, num_images_per_input, strength, guidance_scale, num_inference_steps):
    """
    Processes categories and generates images using the Stable Diffusion pipeline.

    Args:
        pipe: The Stable Diffusion pipeline object.
        input_root (str): Path to the input directory containing categories.
        output_root (str): Path to the output directory for saving generated images.
        prompt (str): Prompt for image generation.
        negative_prompt (str): Negative prompt for image generation.
        num_images_per_input (int): Number of images to generate per input image.
        strength (float): Strength of the image alteration.
        guidance_scale (float): Guidance scale for generation.
        num_inference_steps (int): Number of inference steps.
    """
    categories = ["Defects", "NoDefects"]
    for category in categories:
        input_dir = os.path.join(input_root, category)
        output_dir = output_root
        os.makedirs(output_dir, exist_ok=True)

        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            input_path = os.path.join(input_dir, filename)
            try:
                image = Image.open(input_path).convert("RGB")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
                continue

            # Resize the image to be divisible by 64
            w, h = image.size
            image = image.resize(((w // 64) * 64, (h // 64) * 64))

            # Generate images
            for i in range(num_images_per_input):
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images[0]

                label = "0" if category == "NoDefects" else "1"
                base_name = os.path.splitext(filename)[0]
                out_name = f"{base_name}_{i}_class_{label}.png"

                out_path = os.path.join(output_dir, out_name)
                result.save(out_path)

            print(f"Generated from: {filename} → {num_images_per_input} images")

def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description="Script to generate images using Stable Diffusion and LoRA.")
    parser.add_argument("--input_root", type=str, default="/content/mla_project/images/original", help="Path to the input directory.")
    parser.add_argument("--output_root", type=str, default="/content/mla_project/images/augmented/diffusion/StableDiffusion", help="Path to the output directory.")
    parser.add_argument("--lora_weights_path", type=str, default=None, help="Path to the LoRA weights file.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for image generation.")
    parser.add_argument("--num_images_per_input", type=int, default=1, help="Number of images to generate per input.")
    parser.add_argument("--strength", type=float, default=0.25, help="Strength of the image alteration.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale for generation.")
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Number of inference steps.")
    parser.add_argument("--path_single_image", type=str, default=None, help="Path to a single image to generate from.")
    parser.add_argument("--stable_diffusion_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model to use.")
    args = parser.parse_args()

    # Configuration
    input_root = args.input_root
    output_root = args.output_root
    lora_weights_path = args.lora_weights_path
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    num_images_per_input = args.num_images_per_input
    strength = args.strength
    guidance_scale = args.guidance_scale
    num_inference_steps = args.num_inference_steps
    path_single_image = args.path_single_image
    stable_diffusion_model = args.stable_diffusion_model

    # Load the pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        stable_diffusion_model, torch_dtype=torch.float16
    ).to("cuda")

    if lora_weights_path is not None:
        # Load LoRA weights (Fine-tuned model)
        pipe.load_lora_weights(lora_weights_path)

    # Generate images
    if path_single_image is not None:
        # Generate a single image
        generate_single_image(pipe, path_single_image, output_root, prompt, negative_prompt, num_images_per_input, strength, guidance_scale, num_inference_steps)
    else:
        # Generate images from the input directory
        generate_images(pipe, input_root, output_root, prompt, negative_prompt, num_images_per_input, strength, guidance_scale, num_inference_steps)


if __name__ == "__main__":
    main()