import math
from PIL import Image

from torchvision.transforms import ToTensor,ToPILImage
import torch

PATCH_SIZE       = 14
PATCH_NUM_WIDTH  = 24
PATCH_NUM_HEIGHT = 24
POSITION_EMBEDDING_LENGTH = 1024

MAX_PATCHES      = PATCH_NUM_WIDTH * PATCH_NUM_HEIGHT

TOKEN_LENGTH     = 3 * PATCH_SIZE * PATCH_SIZE

IMAGE_WIDTH      = PATCH_SIZE * PATCH_NUM_WIDTH
IMAGE_HEIGHT     = PATCH_SIZE * PATCH_NUM_HEIGHT

#From LLAVA-UHD
def cal_num_of_slices(origin_image_width, origin_image_height):
    scale = origin_image_width*origin_image_height/(IMAGE_WIDTH*IMAGE_HEIGHT)  
    scale = math.ceil(scale)
    if scale > 6:
        scale = 6
    def factorize(n):
        factors = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append((i/(n/i), i, n // i))
        return factors
    numbers = [1, 2, 3, 4, 5, 6, 7]
    factor_dict = {}
    for num in numbers:
        factor_dict[num] = factorize(num)
    log_origin_ratio = math.log(origin_image_width/origin_image_height)
    available_ratios = []
    if scale<=2:
        available_ratios = factor_dict[scale] + factor_dict[scale + 1]
    else :
        available_ratios = factor_dict[scale-1] + factor_dict[scale]+factor_dict[scale+1]
    min_dif = 1000 
    best_w = 0
    best_h = 0
    for (r,w_slice,h_slice) in available_ratios:
        log_r = math.log(r)
        if min_dif > abs(log_r - log_origin_ratio):
            min_dif = abs(log_r - log_origin_ratio)
            best_w = w_slice
            best_h = h_slice
    
    return best_w,best_h

def split_image(image_file):

    image = Image.open(image_file).convert("RGB")
    width, height = image.size

    # Calculate how to split the image
    num_slices_w, num_slices_h = cal_num_of_slices(width, height)
    # Split the image
    slices = []
    slice_width = width // num_slices_w
    slice_height = height // num_slices_h
    for y in range(0, height - slice_height + 1, slice_height):
        for x in range(0, width - slice_width + 1, slice_width):
            box = (x, y, x + slice_width , y + slice_height)
            slice = image.crop(box)

            # Resize to the target size for CLIP
            resized_slice = slice.resize((IMAGE_WIDTH, IMAGE_HEIGHT)) 
            slices.append(resized_slice)

    return slices , num_slices_h, num_slices_w 

def split_image_tensor(image_tensor):
    #split the image without transforming it to image, just pure tensor operations
    slices = []

    width , height = image_tensor.shape[-1], image_tensor.shape[-2]

    slice_width = IMAGE_WIDTH // PATCH_NUM_WIDTH
    slice_height = IMAGE_HEIGHT // PATCH_NUM_HEIGHT
    num_slices_w, num_slices_h = cal_num_of_slices(width, height)


def get_positional_encoding(max_seq_len, embedding_dim):
    position_encoding = torch.zeros(max_seq_len, embedding_dim)
    position = torch.arange(0, max_seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
    position_encoding[:, 0::2] = torch.sin(position * div_term)
    position_encoding[:, 1::2] = torch.cos(position * div_term)
    return position_encoding