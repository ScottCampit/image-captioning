"""
Text-To-Image example using Stable Diffusion 
"""

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load the Pre-trained Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1.5", 
        torch_dtype=torch.float16
    )
    pipe.to_device("cuda" if torch.cuda.is_available() else "cpu")

    # Inject a prompt for the model
    prompt = "A painting of a cat going to outer space on a unicorn."
    image = pipe(prompt).images[0]

    # Show the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()