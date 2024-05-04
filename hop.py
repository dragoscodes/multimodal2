
    def get_token_embeddings( tokens):
        text = tokens
        input_ids = tokenizer(text).input_ids

        with torch.no_grad():  # Optionally disable gradient calculation
            embeddings = llm.get_input_embeddings()(torch.tensor(input_ids).to(device))

        return embeddings

    def get_positional_encoding(max_seq_len, embedding_dim):
        position_encoding = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding

    def prepare( inputs, images):
        if(images is None):
            return 0
        images = load_images(images)

    def encode_images_positional_encoding( images, padding = True, learned_encoding = True):
        #make sure all images are already preprocessed 
        MAX_LEN = 8

        image_tensors = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].to(device)
        #for the case where there are less than 8 images, add empty tensors
        if(padding):
            for i in range(8-len(images)):
                image_tensors = torch.cat((image_tensors, torch.zeros_like(image_tensors[0]).unsqueeze(0)), dim=0)
        
        with torch.no_grad(): 
            batch_features = vison_model(image_tensors, output_hidden_states=True)
            image_features = batch_features.hidden_states[-1]
            image_features = feature_select(image_features, "patch")
            # Positional Encoding
            max_seq_len = image_features.shape[1]
            pos_encoding = get_positional_encoding(max_seq_len, image_features.shape[-1]).to(device)
            image_features += pos_encoding

        # Learned Encoding
        if learned_encoding:
            image_features += learned_encoding_layer(image_features)

            return projector(image_features)
    
    def images_uhd_positional_encoding( image):
        #lower the image with padding to 
        resized_image = resize_with_padding(image, 336)
        splits , h , w = split_image(image)
        encode_images_positional_encoding(splits)

    def imaged_uhd_arranged( image):
        resized_image = resize_with_padding(image, 336)
        splits , h , w = split_image(image)
        #get the embedding of the tokens "," and "\n" from the llm tokenizer
        tokens = tokenizer.tokenize("\n")
        #get the embedding
        token_embeddings = llm.get_input_embeddings()
        #get the embedding of the tokens
        token_embeddings = token_embeddings(torch.tensor(tokens).to(device))

        encode_images_no_positional_encoding(splits ,padding = False)
        for i in range(h):
            for j in range(w):
                print(f"Image {i*w+j} at position {i},{j}")
    
    def encode_images_no_positional_encoding(image):
        for image in images:
            with torch.no_grad(): 
                batch_features = vison_model(image_tensors, output_hidden_states=True)
                image_features = batch_features.hidden_states[-1]
                image_features = feature_select(image_features, "patch")