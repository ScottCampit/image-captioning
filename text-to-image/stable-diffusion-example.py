"""
Text-To-Image example using Stable Diffusion 
"""

import torch
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load the Pre-trained Stable Diffusion model (note that this is for running on MacOS)
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("mps")
    pipe.enable_attention_slicing()
    
    # Load the Pre-trained Stable Diffusion model (Windows/Linux)
    #from diffusers import StableDiffusionPipeline
    #pipe = StableDiffusionPipeline.from_pretrained(
    #    "runwayml/stable-diffusion-v1.5", 
    #    torch_dtype=torch.float16
    #)
    #pipe.to_device("cuda" if torch.cuda.is_available() else "cpu")

    # Inject a prompt for the model
    prompt = "A painting of a cat flying to outer space on a unicorn with flames behind it."
    image = pipe(prompt).images[0]

    # Show the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()