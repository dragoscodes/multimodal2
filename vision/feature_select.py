def feature_select(image_features, option):
    if option == 'patch':
        image_features = image_features[:, 1:]
    elif option == 'cls_patch':
        image_features = image_features
    else:
        raise ValueError(f'Unexpected select feature: {option}')
    return image_features