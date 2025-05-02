from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt

# === Percorsi ===
img_path = "/content/mla_project/images/original/Defects/Image0.jpg"
output_dir = "/content/mla_project/images/diffusionmodels/ControlNet"
os.makedirs(output_dir, exist_ok=True)

# === 1. Carica immagine e adatta dimensioni ===
init_image = Image.open(img_path).convert("RGB")
w, h = init_image.size
w, h = (w // 64) * 64, (h // 64) * 64
init_image = init_image.resize((w, h))

# === 2. Estrai Canny edge map ===
canny = CannyDetector()
control_image = canny(init_image)

# Salva la Canny map per verifica
canny_path = os.path.join(output_dir, "Image0_canny.png")
control_image.save(canny_path)

# === 3. Carica ControlNet + Stable Diffusion pipeline ===
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
).to("cuda")

# === 4. Prompt e generazione ===
output = pipe(
    prompt = "industrial powder bed surface with fine powder, grayscale, realistic, photo, technical image",
    image = control_image,
    num_inference_steps = 30,
    guidance_scale = 3.5,
    strength = 0.3
).images[0]

# === 5. Salva immagine generata ===
out_path = os.path.join(output_dir, "Image0_aug1.png")
output.save(out_path)

# === 6. Visualizza tutto insieme ===
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(init_image)
axs[0].set_title("Originale")
axs[1].imshow(control_image, cmap="gray")
axs[1].set_title("Mappa Canny")
axs[2].imshow(output)
axs[2].set_title("Generato (ControlNet)")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.show()

print(f"✅ Output salvati in:\n - {out_path}\n - {canny_path}")
