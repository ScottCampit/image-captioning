"""
Image captioning example with FLAVA and Conceptual Captions.
"""

import requests
from io import BytesIO
from PIL import Image

from huggingface_hub import hf_hub_download
import torch
from open_flamingo import create_model_and_transforms
from datasets import load_dataset

def fetch_single_image(url:str) -> Image.Image:
    """Fetch a single image from a URL"""
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        image = None
    return image

if __name__ == "__main__":
    
    # Load data from the Conceptual Captions dataset in the HuggingFace Datasets library
    validation_set = load_dataset("conceptual_captions", split="validation")
    image_url = validation_set['image_url'][0]
    image_caption = validation_set['caption'][0]
    print(f"Original image caption: {image_caption}")
    
    image = fetch_single_image(image_url)
    image.save("./assets/example_image.jpg")
    
    # Load the OpenFlamingo model (3B) and tokenizer
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1
    )    

    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    
    # Preprocess the image and caption
    processed_image = [image_processor(image).unsqueeze(0)]
    processed_image = torch.cat(processed_image, dim=0)
    processed_image = processed_image.unsqueeze(1).unsqueeze(0)
    
    # Note that the initial caption is empty. This is to indicate that the model should generate the caption.
    tokenizer.padding_side = "left"
    processed_caption = tokenizer([f"<image><|endofchunk|>"], return_tensors="pt")
    
    # Run image captioning
    generated_caption = model.generate(
        vision_x=processed_image,
        lang_x=processed_caption["input_ids"],
        attention_mask=processed_caption["attention_mask"],
        max_new_tokens=20,
        num_beams=3,
    )

    # Show the generated caption
    print("Generated text: ", tokenizer.decode(generated_caption[0]))
