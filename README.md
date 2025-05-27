
# 2025/AM04 – Dataset Augmentation for Additive Manufacturing defect detection

This project, titled **"Dataset Augmentation for Additive Manufacturing Defect Detection"** (2025/AM04), was developed for the *Machine Learning in Applications* course at Politecnico di Torino. Our team is identified as group **2025/AM01**.

The aim of this work is to explore and implement data augmentation techniques based on generative models to enhance defect detection in Additive Manufacturing processes.

---

<!-- TABLE OF CONTENTS -->

<div id="top"></div>

## Table of Contents
1. [About The Project](#about-the-project)
   - [Functional Specification](#functional-specification)
   - [Dataset Structure](#dataset-structure)
   - [Technologies and Models Used](#technologies-and-models-used)
   - [Project Structure](#project-structure)
3. [Usage](#usage)
   - [Before Starting](#before-starting)
   - [Classifier on Original Dataset](#classifier-on-original-dataset)
   - [Generative Models](#generative-models)
   - [Metrics](#metrics)
4. [License](#license)
5. [Contributors](#contributors)
6. [References](#references)


---

<!-- ABOUT THE PROJECT -->
## About The Project
<p  align="center">
  <img src="https://i.ibb.co/ffwLDDn/Additive-Manufaturing.png" alt="Additive Manufacturing" width="300" />
</p>

Metal Additive Manufacturing (AM) is a cornerstone of Industry 4.0, offering significant advantages over traditional subtractive manufacturing processes. Despite its potential, AM still faces several quality assurance challenges that limit its adoption in large-scale production. In this context, Computer Vision and Machine Learning (ML) algorithms can support the automation of quality control tasks. However, training accurate classifiers to distinguish between defective and non-defective parts requires a large and balanced dataset—something often unavailable in real production lines. In particular, generating a sufficient number of images showing actual defects is costly and time-consuming due to the expensive nature of the AM process and the difficulty of reproducing specific anomalies on demand.

To address this limitation, we employ generative models to synthesize realistic defect images and augment the available dataset. This data augmentation strategy enhances the training process by improving class balance and increasing the model’s generalization capability, without requiring additional defective parts to be physically produced.

### Functional Specification

Building on this idea, our project focuses on designing and evaluating a complete pipeline for defect detection in AM parts, combining classification and generative techniques. Specifically:

- **Baseline Classification**: We trained a ResNet50 classifier on the original, non-augmented dataset to distinguish between defective and non-defective components.

- **Generative Model Training**: We trained several generative models on the original images to learn the distribution of defects:
   - Variational Autoencoders (VAE)
   - Conditional Variational Autoencoders (CVAE)
   - Generative Adversarial Networks (GANs)
   - Single Image Generative Adversarial Networks (SinGANs)
   - Diffusion Models

- **Synthetic Image Generation**: Each model was used to generate synthetic images representing defective and non-defective parts.

- **Evaluation**:  We assessed the quality and utility of the generated images using multiple metrics: LPIPS (Learned Perceptual Image Patch Similarity), FID (Fréchet Inception Distance), Inception Score (IS), GEN_train / GEN_test (classifier generalization scores).

### Dataset structure

The dataset is composed of two folders:
  - **Defects**: contains 47 images of different layers with one or multiple defects in each of them without labeling.
  - **NoDefects**: contains 33 images of the powder bed without defects.


### Technologies and Models Used

- Python
- PyTorch
- Variational Autoencoders (VAE)
- Conditional Variational Autoencoders (CVAE)
- Generative Adversarial Networks (GANs)
- Single Image Generative Adversarial Networks (SinGANs)
- Diffusion Models
- Evaluation Metrics: LPIPS, GEN-Train/GEN-Test, Direct Analysis of Generated Images, FID, IS

### Project Structure

The project structure is the following:

```
MLA-PRJ-23-PROJECT-AM04/
│
├── generative_models/             # All generative model implementations (training + image generation)
│   ├── Diffusion/                 # Diffusion Models 
│   ├── GANs/                      # Generative Adversarial Networks (GANs)
│   ├── VAEs/                      # Variational Autoencoders (VAEs) and Conditional Variational Autoencoders (CVAEs)
│   └── SinGAN/                    # Single-image GAN
│
├── images/                        # Dataset storage
│   ├── original/                  # Original images
│   └── augmented/                 # Augmented images grouped by method
│
├── metrics/                       # Metric computation scripts and results
│
├── src/                           # Core source code
│   ├── __init__.py
│   ├── data_loader.py             # Dataset loading and preprocessing
│   ├── defect_detection.ipynb     # Main project notebook (entry point)
│   ├── model.py                   # Classifier definition (ResNet)
│   └── train.py                   # Classifier Training logic
│
├── .gitignore                     # Files/folders to be ignored by Git
├── .gitmodules                    # Git submodules (e.g., external repos)
├── LICENSE                        # License file
└── README.md                      # Project documentation (this file)
```

<p align="right">(<a href="#top">back to top</a>)</p>

---

<!-- USAGE EXAMPLES -->
## Usage
The following guide provides step-by-step instructions to navigate and use the notebook `defect_detection.ipynb`, which covers all stages of the project: data preparation, model training, image generation, and evaluation. 

### Before Starting

To get started, clone the repository from GitHub. Next, install all the required dependencies and libraries necessary. 

**Note:** This project requires a machine equipped with a GPU to ensure reasonable training and generation times. We used Google Colab as the primary environment for running the experiments, due to its availability of free GPU resources and ease of use.

### Classifier on Original Dataset

In this section, a classifier is trained on the original dataset to distinguish between images with defects and images without defects, both *without augmentation* and *with basic data augmentation techniques* (such as `RandomHorizontalFlip`, `RandomVerticalFlip`, and `RandomAdjustSharpness`).

The classifier can be trained using either k-fold cross-validation or a standard train-validation split, depending on the parameters set by the user.

### Generative Models

This is the core section of the project, where various Generative Models (listed in the previous section) are implemented and trained to augment the original dataset. All model implementations are contained within the `generative_models` folder and its respective subfolders. Each model has its own dedicated cells for training and for generating new image samples. 

For example, training a Conditional VAE can be performed with the following command:

```
!python /content/mla_project/generative_models/Pytorch-VAE/train_cvae.py \
    --data_dir "/content/mla_project/images/original" \
    --batch_size 4 \
    --max_epoch 100 \
    --latent_size 128 \
    --image_size 512
```
To generate new images using a trained Conditional VAE model, you can run:
```
!python /content/mla_project/generative_models/Pytorch-VAE/generate.py \
    --checkpoint ./checkpoints/model_99.pt \
    --latent_size 128 \
    --image_size 512 \
    --num_images 10 \
    --device cuda \
    --output_dir generated_images

```
*Note: Both training and generation scripts accept various command-line arguments to customize parameters such as batch size, number of epochs, latent size, image size, device, and more.*

### Metrics

The final section of the notebook is dedicated to the evaluation of the generated synthetic images.  
Since visual inspection alone is not sufficient to assess the quality and diversity of the outputs, we have implemented a set of widely-used quantitative metrics: FID (Fréchet Inception Distance), IS (Inception Score), LPIPS (Learned Perceptual Image Patch Similarity) and GEN_train and GEN_test. 

Each metric has a dedicated cell in the notebook.  
To run an evaluation, the user simply needs to specify the **model name** and **experiment ID** corresponding to the generated samples.  

For example, to calculate the LPIPS score on CVAE-generated images from experiment 1:

```
original_dir = "/content/mla_project/images/original/"
generated_dir = "/content/mla_project/images/augmented/"
model = "CVAE"
experiment = "Experiment_1"

!python /content/mla_project/metrics/lpips_score.py --cuda \
    --original_dir {original_dir} \
    --generated_dir {generated_dir} \
    --model {model} \
    --experiment {experiment}
```

<p align="right">(<a href="#top">back to top</a>)</p>

---

<!-- LICENSE -->
## License

This project is distributed under the BSD 3-Clause License, a permissive open-source license that allows you to freely use, modify, and distribute the code, even for commercial purposes, as long as you include the original copyright notice and disclaimers. It does not provide any warranty.

For more details, please refer to the `LICENSE.txt` file.

<p align="right">(<a href="#top">back to top</a>)</p>


---

<!-- Contributors -->
## Contributors

- [Ponzuoli Giacomo](https://github.com/giacomoponzuoli3)
- [Modi Giorgia](https://github.com/GiorgiaModi)
- [Genova Erika](https://github.com/ErikaGenova)
- [Ammirati Marco](https://github.com/TheGoodMark)

<p align="right">(<a href="#top">back to top</a>)</p>

---

<!-- REFERENCES -->
## References

We based part of our work on the implementations provided by the following repositories and papers:

- Tamar Rott Shaham, Tali Dekel, Tomer Michaeli. *"SinGAN: Learning a Generative Model from a Single Natural Image"* (2019)  
  Implementation available at: [https://github.com/tamarott/SinGAN](https://github.com/tamarott/SinGAN)

- Hugging Face Diffusers library: [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)


<p align="right">(<a href="#top">back to top</a>)</p>
