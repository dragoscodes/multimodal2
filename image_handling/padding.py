from PIL import Image
from typing import Tuple
from io import BytesIO
import requests
import torch

def load_images(image_paths):
    images = []
    for image_path in image_paths:
        if image_path.startswith('http') or image_path.startswith('https'):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        images.append(image)
    return images

def load_images_as_tensors(image_paths):
    images = load_images(image_paths)
    image_tensors = []
    for image in images:
        image_tensor = ToTensor()(image).unsqueeze(0)
        image_tensors.append(image_tensor)
    return torch.cat(image_tensors, dim=0)


def calculate_padding_size(image_size, target_size = 336):
    width, height = image_size
    max_dim = max(width, height)
    pad_width = (max_dim - width) // 2
    pad_height = (max_dim - height) // 2
    return (pad_width, pad_height, pad_width, pad_height)


def resize_with_padding(image: Image, target_size: int = 336) -> Image:
    image_size = image.size

    # Calculate aspect ratio
    aspect_ratio = image_size[0] / image_size[1]

    # Resize the image
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Create a new blank image with the target size
    resized_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))

    # Calculate the position to paste the resized image
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2

    # Paste the resized image onto the new blank image with padding
    resized_image.paste(image.resize((new_width, new_height)), (left, top))

    return resized_image

def resize_without_padding(image: Image, target_size: int = 336) -> Image:
    return image.resize((target_size, target_size))

def tensor_resize_with_padding(tensor, target_size):
    """
    Resizes a PyTorch tensor of an image with padding.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    target_size (int): The target size of the image.

    Returns:
    torch.Tensor: The resized image tensor with padding.
    """
    original_size = tensor.shape[1:]
    padding_size = calculate_padding_size(original_size, target_size)
    resized_tensor = torch.nn.functional.pad(tensor, padding_size, value=1)
    resized_tensor = torch.nn.functional.interpolate(resized_tensor.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)
    return resized_tensor

def tensor_resize_without_padding(tensor, target_size):
    """
    Resizes a PyTorch tensor of an image without padding.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    target_size (int): The target size of the image.

    Returns:
    torch.Tensor: The resized image tensor without padding.
    """
    return torch.nn.functional.interpolate(tensor.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of an image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (Tuple[int, int]): The original size of the image.

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    padding_size = calculate_padding_size(original_size, tensor.shape[1])
    unpadded_tensor = tensor[:, padding_size[1]:-padding_size[3], padding_size[0]:-padding_size[2]]
    return unpadded_tensor