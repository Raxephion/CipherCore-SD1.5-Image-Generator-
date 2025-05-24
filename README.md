# CipherCore - Fast Local Stable Diffusion 1.5 Generator (CPU & GPU)

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to the CipherCore Stable Diffusion 1.5 Generator! This user-friendly Gradio web application allows you to effortlessly generate images using various Stable Diffusion 1.5 models. Whether you have local models or prefer popular ones from the Hugging Face Hub, this tool provides a simple interface to unleash your creativity on your CPU or GPU.

*(NOTE: Installer and run file to make it easier for non-engineers and people who don't like fiddling with code and command prompts, coming soon)*

## ✨ Features

*   **Flexible Model Selection:**
    *   Load your own Stable Diffusion 1.5 models (in `diffusers` format) from a local `./checkpoints` directory.
    *   Access a curated list of popular SD1.5 models directly from the Hugging Face Hub (models are downloaded and cached locally on first use).
*   **Device Agnostic:**
    *   Run inference on your **CPU**.
    *   Leverage your **NVIDIA GPU** for significantly faster generation (requires CUDA-enabled PyTorch).
*   **Comprehensive Control:**
    *   **Positive & Negative Prompts:** Guide the AI with detailed descriptions of what you want (and don't want).
    *   **Inference Steps:** Control the number of denoising steps.
    *   **CFG Scale:** Adjust how strongly the image should conform to your prompt.
    *   **Schedulers:** Experiment with different sampling algorithms (Euler, DPM++ 2M, DDPM, LMS).
    *   **Image Sizes:** Choose from standard SD1.5 resolutions, plus a "hire.fix" option (interpreted as 1024x1024).
    *   **Seed Control:** Set a specific seed for reproducible results or use -1 for random generation.
*   **User-Friendly Interface:**
    *   Clean and intuitive Gradio UI.
    *   Organized controls with advanced settings in an accordion for a cleaner look.
    *   Direct image display with download and share options.
*   **Safety First (Note):** The built-in safety checker is **disabled** in this version to allow for maximum creative freedom with custom models. Please be mindful of the content you generate.

## 🚀 Prerequisites

*   **Python:** 3.8 or higher.
*   **Git:** For cloning the repository.
*   **Hardware:**
    *   A modern CPU.
    *   (Recommended for speed) An NVIDIA GPU with CUDA drivers installed if you plan to use GPU acceleration. At least 6-8GB VRAM is recommended for 512x512 generation, more for larger sizes.
*   **Internet Connection:** Required for downloading models from Hugging Face Hub on first use.

## ⚙️ Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Raxephion/CipherCore-SD1.5-Image-Generator-.git
    cd ciphercore-sd1.5-generator
    ```

2.  **Create a Virtual Environment (Recommended):**
    *   Using `venv`:
        ```bash
        python -m venv venv
        # On Windows:
        .\venv\Scripts\activate
        # On macOS/Linux:
        source venv/bin/activate
        ```
    *   Using `conda`:
        ```bash
        conda create -n ciphercore python=3.9
        conda activate ciphercore
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This will install Gradio, PyTorch, Diffusers, and other necessary libraries. PyTorch installation might take some time and will attempt to install a version compatible with your system's CUDA if available.)*

4.  **Prepare Local Models (Optional):**
    *   Create a directory named `checkpoints` in the root of the project.
    *   Place your Stable Diffusion 1.5 models (in `diffusers` format – meaning each model is a folder containing files like `model_index.json`, `unet/`, `vae/`, etc.) inside the `checkpoints` directory.
        Example structure:
        ```
        ciphercore-sd1.5-generator/
        ├── checkpoints/
        │   ├── my-custom-model-1/
        │   │   ├── model_index.json
        │   │   ├── unet/
        │   │   └── ...
        │   └── another-local-model/
        │       └── ...
        ├── app.py
        └── ...
        ```

## ▶️ Running the Application

Once the setup is complete, launch the Gradio web UI:

```bash
python app.py
