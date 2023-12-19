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
    question = validation_set['question'][0]
    print(f"Original question: {question}")
    
    # Load BLIP Model and Processor
    processor = BlipProcessor.from_pretrained("ybelkada/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "ybelkada/blip-vqa-base"
    )
    
    # Inference with BLIP
    inputs = processor(image, question, return_tensors="pt")
    outputs = model.generate(**inputs)
    print(f"Generated answer: {processor.tokenizer.decode(outputs[0])}")
    print(f"Answer in TextQA dataset: {validation_set['answers'][0]}")