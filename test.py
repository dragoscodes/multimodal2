import torch
from transformers import PreTrainedModel
from loader.model_loader import load_vision_model, load_llm
from vision.projector import load_vision_projector
from vision.feature_select import feature_select
from vision.learned_encoding import load_learned_encoding
from image_handling.padding import resize_with_padding, load_images
from image_handling.slice import split_image
from transformers import BitsAndBytesConfig
import math
import requests
from PIL import Image
from io import BytesIO


def get_positional_encoding(max_seq_len, embedding_dim):
    position_encoding = torch.zeros(max_seq_len, embedding_dim)
    position = torch.arange(0, max_seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
    position_encoding[:, 0::2] = torch.sin(position * div_term)
    position_encoding[:, 1::2] = torch.cos(position * div_term)
    return position_encoding

def imaged_uhd_arranged(image):
    #resized_image = resize_with_padding(image, 336)
    splits , h , w = split_image(image)
    #save the splits
    i=0
    for split in splits:
        i=i+1
        split.save(f"split_{i}.jpg")
    for i in range(h):
        for j in range(w):
            print(f"Image {i*w+j} at position {i},{j}")
    
imaged_uhd_arranged(image = "img.jpg")