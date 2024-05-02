def resize_with_padding(image_path, target_size = 336):
    image = Image.open(image_path).convert("RGB")
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




def encode_images(image_paths, image_processor, vision_tower):
    image_features_list = []
    for image_path in image_paths:
        image = resize_with_padding(image_path, 336)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to("cpu")
        image_features = vision_tower(image_tensor, output_hidden_states=True)
        image_features_list.append(image_features.hidden_states[-1])
    return torch.cat(image_features_list, dim=0)