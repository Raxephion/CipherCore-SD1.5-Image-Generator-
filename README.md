Basic Stable Diffusion 1.5 Gradio App
A simple Gradio web application for generating images using Stable Diffusion 1.5 models. Users can select models from a local ./checkpoints directory or choose from a list of popular models on the Hugging Face Hub. The app supports CPU or GPU inference (if a compatible GPU is available) and allows users to configure various generation parameters.
Features
Model Selection: Choose from locally stored Stable Diffusion 1.5 models (place model subdirectories in ./checkpoints/) or a curated list of popular models from the Hugging Face Hub (downloads if not cached).
Device Selection: Select to run inference on CPU or GPU (if CUDA-enabled PyTorch is installed and a compatible GPU is available).
Prompt Control: Enter positive and negative prompts to guide image generation.
Parameter Adjustment: Configure inference steps, CFG scale, scheduler, image size, and random seed for reproducibility.
"Hire.fix" Option: Includes a "hire.fix" option which is interpreted as generating a 1024x1024 image in this basic implementation.
Image Output: Displays the generated image with options to share and download.
Prerequisites
Python 3.7+
Git (for cloning)
A compatible CPU or NVIDIA GPU with CUDA drivers installed (for GPU inference).
An internet connection (for downloading models from the Hugging Face Hub if not available locally).
Setup
Clone the repository:
git clone <your-repo-url>
cd stable-diffusion-gradio
Create and activate a virtual environment:
Using venv:
python -m venv venv
On Windows
.\venv\Scripts\activate
On macOS/Linux
source venv/bin/activate
Using conda:
conda create -n stablediffusion python=3.9 # Or your preferred Python version
conda activate stablediffusion
Install dependencies:
pip install -r requirements.txt
Local Models (Optional):
If you have Stable Diffusion 1.5 models in the diffusers format locally, create a checkpoints directory in the project root and place each model in its own subdirectory within checkpoints.
Usage
Run the Gradio application from your terminal:
python stable_diffusion_app.py
This will launch a web interface in your default browser. You can then:
Select a model from the "Select Model" dropdown.
Choose "CPU" or "GPU" from the "Select Device" dropdown.
Enter your desired prompt in the "Prompt" textbox.
Optionally, enter a negative prompt.
Adjust the "Inference Steps" and "CFG Scale" sliders.
Select a "Scheduler" and "Image Size" (including "hire.fix" for 1024x1024).
Enter a seed for reproducible results, or leave as -1 for a random seed.
Click the "Generate Image" button.
The generated image will be displayed in the output area.
Important Notes
Local Models: Ensure your local models are in the diffusers format (typically a directory with configuration files and .bin or .safetensors weights).
Hugging Face Hub: If a selected Hub model is not already cached, it will be downloaded.
GPU Usage: GPU inference requires PyTorch with CUDA support to be correctly installed and a compatible NVIDIA GPU.
Memory: Generating high-resolution images or using complex models can be memory-intensive. You may encounter "Out of Memory" errors on systems with limited resources. Try reducing image size or inference steps.
Safety Checker: The safety checker has been intentionally removed in this basic implementation. Use with caution and be aware of the potential for generating NSFW content.
Contributing
Feel free to fork the project, make improvements, and submit pull requests.
