"""
Visual Question Answering example with BLIP and TextVQA.
"""

import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from datasets import load_dataset

if __name__ == "__main__":
    # Load data from the TextVQA dataset in the HuggingFace Datasets library
    validation_set = load_dataset("textvqa", split="validation")
    image = validation_set['image'][0]
    print(image)
    question = validation_set['question'][0]
    
    # Load BLIP Model and Processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("ybelkada/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "ybelkada/blip-vqa-base", 
        torch_dtype=torch.float16
    ).to(device)
    
    # Inference with BLIP
    inputs = processor(image, question, return_tensors="pt").to(device, torch.float16)
    outputs = model(**inputs)
    print(outputs)