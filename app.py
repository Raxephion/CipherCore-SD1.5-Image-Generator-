# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 2024

@author: raxephion
Basic Stable Diffusion 1.5 Gradio App with local/Hub models and CPU/GPU selection 

"""

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
# Import commonly used schedulers
from diffusers import DDPMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler
import os
from PIL import Image
import time # Optional: for timing generation
from huggingface_hub import HfFolder, login # Optional: for logging in

# --- Configuration ---
MODELS_DIR = "checkpoints"
# Standard SD 1.5 sizes (multiples of 64 are generally safe)
# Models are primarily trained on 512x512. Other sizes might show artifacts.
# Added 'hire.fix' as an option, interpreted as 1024x1024 in this script
SUPPORTED_SD15_SIZES = ["512x512", "768x512", "512x768", "768x768", "1024x768", "768x1024", "1024x1024", "hire.fix"]


# Mapping of friendly scheduler names to their diffusers classes
SCHEDULER_MAP = {
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DDPM": DDPMScheduler,
    "LMS": LMSDiscreteScheduler,
    # Add more as needed from diffusers.schedulers (make sure they are imported)
}
DEFAULT_SCHEDULER = "Euler" # Default scheduler on startup

# List of popular Stable Diffusion 1.5 models on the Hugging Face Hub
# You can add more Hub model IDs here
DEFAULT_HUB_MODELS = [
    "runwayml/stable-diffusion-v1-5",
    "SG161222/Realistic_Vision_V6.0_B1_noVAE", # Example popular 1.5 model
    # "SG161222/RealVisXL_V5.0_Lightning", # Note: RealVisXL is SDXL, might not work well or at all with SD1.5 pipeline/schedulers (Removed as it might confuse users for an SD1.5 app)
    "nitrosocke/Ghibli-Diffusion",
    "danyloylo/sd1.5-ghibli-style-05",
    "Bilal326/SD_1.5_DragonWarriorV2"
    # "CompVis/stable-diffusion-v1-4", # Example SD 1.4 model (might behave slightly differently)
    # Add other diffusers-compatible SD1.5 models here
]

# --- Determine available devices and set up options ---
AVAILABLE_DEVICES = ["CPU"]
if torch.cuda.is_available():
    AVAILABLE_DEVICES.append("GPU")
    print(f"CUDA available. Found {torch.cuda.device_count()} GPU(s).")
    if torch.cuda.device_count() > 0:
        print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Running on CPU.")

# Default device preference: GPU if available, else CPU
DEFAULT_DEVICE = "GPU" if "GPU" in AVAILABLE_DEVICES else "CPU"


# --- Global state for the loaded pipeline ---
# We'll load the pipeline once and keep it in memory
current_pipeline = None
current_model_id = None # Keep track of the currently loaded model identifier
current_device_loaded = None # Keep track of the device the pipeline is currently on


# --- Helper function to list available local models ---
def list_local_models(models_dir):
    """Scans the specified directory for subdirectories (potential local diffusers models)."""
    # Create the models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")
        return [] # No models if directory was just created

    # Get absolute path for more robust comparison later
    abs_models_dir = os.path.abspath(models_dir)

    # List subdirectories (potential models)
    # Return their full relative path from the script location
    local_models = [os.path.join(models_dir, d) for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))]

    return local_models


# --- Image Generation Function ---
# Added 'selected_device_str' parameter
def generate_image(model_identifier, selected_device_str, prompt, negative_prompt, steps, cfg_scale, scheduler_name, size, seed):
    """Generates an image using the selected model and parameters on the chosen device."""
    global current_pipeline, current_model_id, current_device_loaded, SCHEDULER_MAP

    if not model_identifier or model_identifier == "No models found":
        raise gr.Error(f"No model selected or available. Please add models to '{MODELS_DIR}' or ensure Hub IDs are correct in the script.")
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    # Map selected device string to PyTorch device string
    device_to_use = "cuda" if selected_device_str == "GPU" and "GPU" in AVAILABLE_DEVICES else "cpu"
    # If GPU was selected but not available, raise an error
    if selected_device_str == "GPU" and device_to_use == "cpu":
         raise gr.Error("GPU selected but CUDA is not available. Please select CPU or ensure PyTorch with CUDA is installed correctly.")

    # Determine dtype based on the actual device being used
    # Note: fp16 is generally faster and uses less VRAM on compatible GPUs
    dtype_to_use = torch.float16 if device_to_use == "cuda" else torch.float32

    print(f"Attempting generation on device: {device_to_use}, using dtype: {dtype_to_use}")

    # 1. Load Model if necessary
    # Check if the requested model OR the device has changed
    if current_pipeline is None or current_model_id != model_identifier or (current_device_loaded is not None and str(current_device_loaded) != device_to_use):
        print(f"Loading model: {model_identifier} onto {device_to_use}...")
        # Clear previous pipeline to potentially free memory *before* loading the new one
        if current_pipeline is not None:
             print(f"Unloading previous model '{current_model_id}' from {current_device_loaded}...")
             # Move pipeline to CPU before deleting if it was on GPU, might help with freeing VRAM
             if current_device_loaded == torch.device("cuda"):
                  current_pipeline.to("cpu")
             del current_pipeline
             current_pipeline = None # Set to None immediately
             # Attempt to clear CUDA cache if using GPU
             if current_device_loaded == torch.device("cuda"): # Clear cache of the *previous* device
                 try:
                     torch.cuda.empty_cache()
                     print("Cleared CUDA cache.")
                 except Exception as cache_e:
                     print(f"Error clearing CUDA cache: {cache_e}") # Don't stop if cache clearing fails

        # Ensure the device is actually available if not CPU
        if device_to_use == "cuda":
             if not torch.cuda.is_available():
                  raise gr.Error("CUDA selected but not available. Please install PyTorch with CUDA support or select CPU.")

        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_identifier,
                torch_dtype=dtype_to_use,
                safety_checker=None, # <<< REMOVED SAFETY CHECKER HERE <<<
            )
            pipeline = pipeline.to(device_to_use)
            current_pipeline = pipeline
            current_model_id = model_identifier
            current_device_loaded = torch.device(device_to_use)

            # Basic check for SD1.x architecture
            if not (hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'config') and
                    hasattr(pipeline.unet.config, 'cross_attention_dim') and
                    pipeline.unet.config.cross_attention_dim == 768): # SD1.x typically has 768
                 warning_msg = (f"Warning: Loaded model '{model_identifier}' might not be a standard SD 1.5 model "
                                f"(expected UNet cross_attention_dim 768, found {getattr(pipeline.unet.config, 'cross_attention_dim', 'N/A')}). "
                                "Results may be unexpected or generation might fail.")
                 print(warning_msg)
                 gr.Warning(warning_msg)


            print(f"Model '{model_identifier}' loaded successfully on {current_device_loaded}.")

        except Exception as e:
            current_pipeline = None
            current_model_id = None
            current_device_loaded = None
            print(f"Error loading model '{model_identifier}': {e}")
            error_message_lower = str(e).lower()
            if "cannot find requested files" in error_message_lower or "404 client error" in error_message_lower or "no such file or directory" in error_message_lower:
                 raise gr.Error(f"Model '{model_identifier}' not found. Check name/path.")
            elif "checkpointsnotfounderror" in error_message_lower:
                 raise gr.Error(f"No valid diffusers model at '{model_identifier}'.")
            elif "out of memory" in error_message_lower:
                 raise gr.Error(f"Out of Memory (OOM) loading model. Try a lighter model or check system resources.")
            elif "cusolver64" in error_message_lower or "cuda driver version" in error_message_lower:
                 raise gr.Error(f"CUDA/GPU Driver Error: {e}. Check drivers/PyTorch install.")
            elif "safetensors_rust.safetensorserror" in error_message_lower or "oserror: cannot load" in error_message_lower:
                 raise gr.Error(f"Model file error for '{model_identifier}': {e}. Files might be corrupt.")
            else:
                raise gr.Error(f"Failed to load model '{model_identifier}': {e}")

    # 2. Configure Scheduler
    selected_scheduler_class = SCHEDULER_MAP.get(scheduler_name)
    if selected_scheduler_class is None:
         print(f"Warning: Unknown scheduler '{scheduler_name}'. Using default: {DEFAULT_SCHEDULER}.")
         selected_scheduler_class = SCHEDULER_MAP[DEFAULT_SCHEDULER]
    try:
        current_pipeline.scheduler = selected_scheduler_class.from_config(current_pipeline.scheduler.config)
        print(f"Scheduler set to: {scheduler_name}")
    except Exception as e:
        print(f"Error setting scheduler '{scheduler_name}': {e}")
        raise gr.Error(f"Failed to configure scheduler '{scheduler_name}': {e}")

    # 3. Parse Image Size
    width, height = 512, 512
    if size.lower() == "hire.fix":
        width, height = 1024, 1024
        print(f"Interpreting 'hire.fix' size as {width}x{height}")
    else:
        try:
            width, height = map(int, size.split('x'))
        except ValueError:
            raise gr.Error(f"Invalid size: '{size}'. Use 'WidthxHeight' or 'hire.fix'.")
        except Exception as e:
             raise gr.Error(f"Error parsing size '{size}': {e}")

    if width % 64 != 0 or height % 64 != 0:
         warning_msg_size = f"Warning: Size {width}x{height} not a multiple of 64. May cause issues."
         print(warning_msg_size)
         gr.Warning(warning_msg_size)

    # 4. Set Seed Generator
    generator = None
    generator_device = current_pipeline.device if current_pipeline else device_to_use
    if seed != -1:
        try:
            generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        except Exception as e:
             print(f"Error setting seed generator: {e}")

    # 5. Generate Image
    print(f"Generating: Prompt='{prompt[:80]}...', NegPrompt='{negative_prompt[:80]}...', Steps={int(steps)}, CFG={float(cfg_scale)}, Size={width}x{height}, Scheduler={scheduler_name}, Seed={int(seed)}")
    start_time = time.time()

    try:
        output = current_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg_scale),
            width=width,
            height=height,
            generator=generator,
        )
        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")
        generated_image = output.images[0]
        return generated_image

    except gr.Error as e:
         raise e
    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        error_message_lower = str(e).lower()
        if "size must be a multiple of" in error_message_lower or "invalid dimensions" in error_message_lower or "shape mismatch" in error_message_lower:
             raise gr.Error(f"Image generation failed - Invalid size '{width}x{height}' for model: {e}")
        elif "out of memory" in error_message_lower or "cuda out of memory" in error_message_lower:
             raise gr.Error(f"Out of Memory (OOM) during generation. Try smaller size/steps. {e}")
        elif "runtimeerror" in error_message_lower:
             raise gr.Error(f"Runtime Error during generation: {e}")
        else:
             raise gr.Error(f"Image generation failed: {e}")

# --- Gradio Interface ---
local_models = list_local_models(MODELS_DIR)
model_choices = local_models + DEFAULT_HUB_MODELS

if not model_choices:
    initial_model_choices = ["No models found"]
    initial_default_model = "No models found"
    model_dropdown_interactive = False
    print(f"\n--- IMPORTANT ---")
    print(f"No local models in '{MODELS_DIR}' and no default Hub models listed.")
    print(f"Place diffusers SD 1.5 models in '{os.path.abspath(MODELS_DIR)}' or add Hub IDs to DEFAULT_HUB_MODELS in script.")
    print(f"-----------------\n")
else:
    initial_model_choices = model_choices
    if "runwayml/stable-diffusion-v1-5" in model_choices:
         initial_default_model = "runwayml/stable-diffusion-v1-5"
    elif local_models:
         initial_default_model = local_models[0]
    else:
         initial_default_model = model_choices[0]
    model_dropdown_interactive = True

scheduler_choices = list(SCHEDULER_MAP.keys())

with gr.Blocks(theme=gr.themes.Soft()) as demo: # Added a soft theme for better aesthetics
    gr.Markdown(
        """
        # CipherCore Stable Diffusion 1.5 Generator
        Create images with Stable Diffusion 1.5. Supports local models from `./checkpoints`
        and select models from Hugging Face Hub.
        _Note: 'hire.fix' size option generates at 1024x1024._
        """
    )

    with gr.Row():
        with gr.Column(scale=2): # Give more space to controls
            model_dropdown = gr.Dropdown(
                choices=initial_model_choices,
                value=initial_default_model,
                label="Select Model (Local or Hub)",
                interactive=model_dropdown_interactive,
            )
            device_dropdown = gr.Dropdown(
                choices=AVAILABLE_DEVICES,
                value=DEFAULT_DEVICE,
                label="Processing Device",
                interactive=len(AVAILABLE_DEVICES) > 1,
            )
            prompt_input = gr.Textbox(label="Positive Prompt", placeholder="e.g., a majestic lion in a vibrant jungle, photorealistic", lines=3, autofocus=True) # Autofocus on prompt
            negative_prompt_input = gr.Textbox(label="Negative Prompt (Optional)", placeholder="e.g., blurry, low quality, deformed, watermark", lines=2)

            with gr.Accordion("Advanced Settings", open=False): # Keep advanced settings initially closed
                with gr.Row():
                    steps_slider = gr.Slider(minimum=5, maximum=150, value=30, label="Inference Steps", step=1)
                    cfg_slider = gr.Slider(minimum=1.0, maximum=30.0, value=7.5, label="CFG Scale", step=0.1) # Increased max CFG
                with gr.Row():
                     scheduler_dropdown = gr.Dropdown(
                        choices=scheduler_choices,
                        value=DEFAULT_SCHEDULER,
                        label="Scheduler"
                    )
                     size_dropdown = gr.Dropdown(
                        choices=SUPPORTED_SD15_SIZES,
                        value="512x512",
                        label="Image Size"
                    )
                seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0) # precision=0 for integer

            generate_button = gr.Button("✨ Generate Image ✨", variant="primary", scale=1) # Added emojis

        with gr.Column(scale=3): # Give more space to image
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                height=768, # Slightly larger preview if possible
                width=768,
                show_share_button=True,
                show_download_button=True,
                interactive=False # Output image is not interactive
            )
            # Add a status display (optional, but good for user feedback during long loads)
            # status_display = gr.Textbox(label="Status", interactive=False, lines=1)

    # Link button click to generation function
    # The `api_name` parameter allows calling this function via API if app is deployed with --share or similar
    generate_button.click(
        fn=generate_image,
        inputs=[
            model_dropdown,
            device_dropdown,
            prompt_input,
            negative_prompt_input,
            steps_slider,
            cfg_slider,
            scheduler_dropdown,
            size_dropdown,
            seed_input
        ],
        outputs=[output_image],
        api_name="generate" # Optional: For API access
    )

if __name__ == "__main__":
    if not model_choices:
         print(f"\n!!! No models available. Gradio app might not function correctly. Add models to '{MODELS_DIR}' or script. !!!")

    print("Launching CipherCore Stable Diffusion 1.5 Generator...")
    demo.launch(show_error=True, inbrowser=True) # Launch in browser by default
