
<h2>2025/AM04 – Dataset Augmentation for
Additive Manufacturing defect detection</h3>


---

<!-- TABLE OF CONTENTS -->

<div id="top"></div>

### Table of Contents
1. [About The Project](#about-the-project)
   - [Dataset Structure](#dataset-structure)
   - [Technologies and Models Used](#technologies-and-models-used)
   - [Functional Specification](#functional-specification)
2. [Guide](#guide)
   - [Notebook Structure](#notebook-structure)
   - [Usage](#usage)
      - [Before Starting](#before-starting)
      - [Classifier on Original Dataset](#classifier-on-original-dataset)
      - [Generative Models](#generative-models)
      - [Metrics](#metrics)
3. [License](#license)
4. [Contacts](#contacts)
5. [References](#references)



---

<!-- ABOUT THE PROJECT -->
### About The Project
<p  align="center">
  <img src="https://i.ibb.co/ffwLDDn/Additive-Manufaturing.png" alt="Additive Manufacturing" width="300" />
</p>

Metal Additive Manufacturing (AM) is a pillar of the Industry 4.0, with many attractive advantages compared to traditional subtractive fabrication technologies. However, there are many quality issues that can be an obstacle for mass production. In this context, the use of Generative Models algorithms have a very important role. Nonetheless, they are up to this date limited by the scarcity of data for the training, as well as by the difficulty of accessing and integrating the AM process data throughout the fabrication. To tackle this problem, an generative model algorithm is required to increase the number of images available to train the classifier on the generated dataset. 

The category of defects addressed are the following:

1. **Holes**: localised lacks of metallic powder that create small dark areas in the powder bed image. They are generally due to a bad regulation of the powder dosing factor, leading to local lacks of powder.
2. **Spattering**: droplets of melted metal ejected from the melt pool and landed in the surroundings.
3. **Incandescence**: high-intensity areas in the powder bed layer. It is generally a consequence of the inability of the melt pool to cool down correctly, due to an excess of laser energy power.
4. **Horizontal defects**: dark horizontal lines in the layer image caused by geometric imperfection of the piece that leads to the incorrect spreading of the metallic powder.
5. **Vertical defects**: vertical undulation of the powder bed along the direction of the recoater’s path, consisting in alternated dark and light lines. The origin is either a mechanical defect of the recoater’s surface or a mechanical interference between the object and the recoater.

In the following image is reported an example of the defects.
<p align="center">
  <img src="https://i.ibb.co/0yPGZL7r/Defects-Examples.png" alt="Examples of Defects" width=600 />
</p>


#### Dataset structure

The dataset is composed of two folders:
  - **Defects**: contains a set of images with several defects like holes, splattering, etc. They consist of 47 images of different layers with one or multiple defects in each of them without labeling
  - **NoDefects**: contains plain images of the powder bed without defects. They consists of 33 images without defects


#### Technologies and Models Used

- Python
- PyTorch
- Variational Autoencoders (VAE)
- Conditional VAEs (CVAE)
- Generative Adversarial Networks (GANs)
- SinGAN
- Diffusion Models
- Evaluation Metrics: FID, L-PIPS, IS, GEN with training set and GEN with test set


#### Functional Specification

This application focuses on generating synthetic datasets for the Additive Manufacturing domain.  
Its main purpose is to overcome the scarcity of defective samples by using generative models such as Variational Autoencoders (VAE), Conditional VAEs (CVAE), Generative Adversarial Networks (GANs), and Diffusion Models.

The generated images are used to train classifiers for defect detection, improving performance despite the limited size of the original dataset.

<p align="right">(<a href="#top">back to top</a>)</p>

---


<!-- GUIDE -->
### Guide

The following guide provides step-by-step instructions to navigate and use the Colab notebook, which covers all stages of the project: data preparation, model training, image generation, and evaluation.  
The notebook is located in the `src` folder, called `defect_detection.ipynb`.

#### Notebook Structure

The notebook is organized into the following main sections:

- **Before Starting**
  - Clone the repository
  - Install dependencies
  - Import libraries

- **Classifier on Original Dataset**
  - Training without augmentation
  - Training with basic augmentation

- **Generative Models**
  - SinGAN
  - GANs
  - Diffusion Models
  - Conditional VAE (CVAE)
  - VAE

- **Metrics**
  - L-PIPS (Learned Perceptual Image Patch Similarity)
  - FID (Fréchet Inception Distance)
  - IS (Inception Score)
  - GEN (evaluated on training and test sets)


<!-- USAGE EXAMPLES -->
#### Usage

This section explains the functionality of each part of the Colab notebook described above.

##### Before Starting

To get started, clone the repository from GitHub.  
Next, install all the required dependencies and libraries necessary to run the project smoothly.

##### Classifier on Original Dataset

In this section, the classifier is trained using the original dataset, both **without augmentation** and **with basic data augmentation techniques** (such as `transforms.RandomHorizontalFlip()`, `transforms.RandomVerticalFlip()`, and `transforms.RandomAdjustSharpness(sharpness_factor=2)`).

The classifier can be trained using either **k-fold cross-validation** or a **standard train-validation split**, depending on the parameters set by the user.

##### Generative Models

This is the core section of the project, where various Generative Models are implemented and trained to augment the original dataset. The goal is to generate realistic synthetic images that can enhance the training of classifiers, especially when working with limited data.

The models available in this notebook include:
- **VAE** (Variational Autoencoder)
- **CVAE** (Conditional Variational Autoencoder)
- **GANs** (Generative Adversarial Networks)
- **SinGAN** (Single Image GAN)
- **Diffusion Models**

Each model has its own dedicated cells for training and for generating new image samples.  
Once generated, the synthetic images are automatically saved in the following directory structure:  
`images/augmented/<model_name>/<experiment_num>`

This modular setup allows easy comparison between the performance of different generative approaches and facilitates the integration of new generated data into the classification pipeline.

##### Metrics

The final section of the notebook is dedicated to the evaluation of the generated synthetic images.  
Since visual inspection alone is not sufficient to assess the quality and diversity of the outputs, we have implemented a set of widely-used quantitative metrics:

- **FID (Fréchet Inception Distance):** Measures the similarity between the distribution of generated images and real images.
- **IS (Inception Score):** Evaluates the quality and diversity of generated images based on how well they can be classified.
- **L-PIPS (Learned Perceptual Image Patch Similarity):** Computes perceptual similarity between images using deep network features.
- **GEN_train and GEN_test:** Custom metrics designed to evaluate the contribution of generated images to the training and generalization performance of the classifier.

Each metric has a dedicated cell in the notebook.  
To run an evaluation, the user simply needs to specify the **model name** and **experiment ID** corresponding to the generated samples.  
This setup enables flexible and repeatable assessment across different models and generations, helping identify the most effective data augmentation strategies.


<p align="right">(<a href="#top">back to top</a>)</p>


---

<!-- LICENSE -->
### License

This project is distributed under the BSD 3-Clause License, a permissive open-source license that allows you to freely use, modify, and distribute the code, even for commercial purposes, as long as you include the original copyright notice and disclaimers. It does not provide any warranty.

For more details, please refer to the `LICENSE.txt` file.

<p align="right">(<a href="#top">back to top</a>)</p>


---

<!-- CONTACTS -->
### Contacts

Ponzuoli Giacomo - s332271@studenti.polito.it
Modi Giorgia - s330519@studenti.polito.it
Genova Erika - s332044@studenti.polito.it
Ammirati Marco - s300269@studenti.polito.it

<p align="right">(<a href="#top">back to top</a>)</p>

---

<!-- REFERENCES -->
### References

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
