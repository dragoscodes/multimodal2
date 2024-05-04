import torch
from transformers import PreTrainedModel
from loader.model_loader import load_vision_model, load_llm
from vision.projector import load_vision_projector
from vision.feature_select import feature_select
from vision.learned_encoding import load_learned_positional
from image_handling.padding import resize_with_padding, load_images
from image_handling.slice import split_image
from transformers import BitsAndBytesConfig
import math
import requests
from PIL import Image
from io import BytesIO

# class LeMultiModalConfig:
#     def __init__(self, llm, vision_model, ):
#         self.image_processing_method = image_processing_method
#         self.learning_rate = learning_rate
#         # ... other parameters


class LeMultiModal(PreTrainedModel):
    #Usually they do some config here, but I'm skipping that for now

    def __init__(self, config):
        super().__init__(config)
        self.max_len = 8
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.vision_model , self.image_processor = load_vision_model("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", device = self.device )
        self.llm, self.tokenizer = load_llm("llama3/8B-instruct", device = self.device, quantization_config = self.quantization_config)
        self.vision_projector = load_vision_projector()
        self.llm_dim = self.llm.config.hidden_size
        self.vision_dim = self.vision_model.config.hidden_size
        self.learned_positional = load_learned_positional(self.max_len, )
        self.uhd_sepparators = self.get_token_embeddings(["\n", ","])

    def get_token_embeddings(self, text):
        input_ids = self.tokenizer(text).input_ids

        with torch.no_grad():  # Optionally disable gradient calculation
            embeddings = self.llm.get_input_embeddings()(torch.tensor(input_ids).to(self.device))

        return embeddings

    def get_positional_encoding(max_seq_len, embedding_dim):
        position_encoding = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding

    def processs(self, image, text):
        #Supports just 1 image for now
        
        if "<image>" not in text:
            #the embedding is just the text
            new_embeddings = self.get_token_embeddings(text)
        else:
            assert text.count("<image>") == 1
            new_embeddings = self.encode_images_no_positional_encoding(image)
            before, after = text.split("<image>")
            if len(before) > 0:
                new_embeddings = torch.cat((self.get_token_embeddings(before), new_embeddings), dim=0)
            if len(after) > 0:
                new_embeddings = torch.cat((new_embeddings, self.get_token_embeddings(after)), dim=0)

        #run the embeddings through the llm and return the result in clear text
        with torch.no_grad():
            output = self.llm(new_embeddings.unsqueeze(0))
            return self.tokenizer.decode(output[0])

    def encode_images_positional_encoding(self, images, padding = True, sinusoidal_encoding = True, learned_encoding = False):
        MAX_LEN = 8

        image_tensors = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values'].to(self.device)
        #for the case where there are less than 8 images, add empty tensors
        if(padding):
            for i in range(MAX_LEN-len(images)):
                image_tensors = torch.cat((image_tensors, torch.zeros_like(image_tensors[0]).unsqueeze(0)), dim=0)
        
        with torch.no_grad(): 
            batch_features = self.vison_model(image_tensors, output_hidden_states=True)
            image_features = batch_features.hidden_states[-1]
            image_features = feature_select(image_features, "patch")
            # Positional Encoding
            if(sinusoidal_encoding):
                max_seq_len = image_features.shape[1]
                pos_encoding = self.get_positional_encoding(max_seq_len, image_features.shape[-1]).to(self.device)
                image_features += pos_encoding

        # Learned Positional Encoding
        if learned_encoding:
            image_features += self.learned_encoding_layer(image_features)

        return self.vision_projector(image_features)
    
    def images_uhd_positional_encoding(self, image):
        #lower the image with padding to 
        resized_image = resize_with_padding(image, 336)
        splits , h , w = split_image(image)
        self.encode_images_positional_encoding(splits)

    def imaged_uhd_arranged(self, image):
        resized_image = resize_with_padding(image, 336)
        splits , h , w = split_image(image)

        embeddings = self.encode_images_no_positional_encoding(splits)
        new_embeddings = []
        for i in range(h):
            for j in range(w):
                new_embeddings.append(embeddings[i*w+j])
                new_embeddings.append(self.uhd_sepparators[1])
            new_embeddings.append(self.uhd_sepparators[0])
        
        return new_embeddings
                
    
    def encode_images_no_positional_encoding(self, image_tensors):
        with torch.no_grad(): 
            batch_features = self.vison_model(image_tensors, output_hidden_states=True)
            image_features = batch_features.hidden_states[-1]
            image_features = feature_select(image_features, "patch")
        return self.vision_projector(image_features)
    

    # def forward_1image(self, image, text):
    #     #Supports just 1 image for now
    #     if "<image>" not in text:
    #         #the embedding is just the text
    #     else:
    #         assert text.count("<image>") == 1
    #         before, after = text.split("<image>")
    #         if len(before) > 0:
    #             #embedding of tokenized(before)
    #             #add embedding of image
    #         if len(after) > 0:
    #             #embedding of tokenized(after)



###
#            image_features = self.vison_model(image_tensor, output_hidden_states=True)
#            image_features = image_features.hidden_states[-1]
#            image_features = feature_select(image_features, "patch")
#             # Positional Encoding (assuming sinusoidals)
#            max_seq_len = image_features.shape[1] 
#            pos_encoding = get_positional_encoding(max_seq_len, batch_features.shape[-1])
#            pos_encoding = pos_encoding.to(self.device)
#            batch_features += pos_encoding
#
#            # Learned Encoding
#            learned_encoding = self.learned_encoding_layer(batch_features)
#            batch_features += learned_encoding 
#
#            image_features = projector(batch_features)
#
#            image_features = self.vision_projector(image_features)
#            image_features_list.append(image_features)
###

    def generate(
        self,
        image_embeds,
        prompt,
        tokenizer,
        max_new_tokens=128,
        **kwargs,
    ):
        generate_config = {
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.bos_token_id,
            "max_new_tokens": max_new_tokens,
            **kwargs,
        }

        with torch.no_grad():
            inputs_embeds = self.input_embeds(prompt, image_embeds, tokenizer)
            output_ids = self.text_model.generate(
                inputs_embeds=inputs_embeds, **generate_config
            )

        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def answer_question(
        self,
        image_embeds,
        question,
        tokenizer,
        chat_history="",
        result_queue=None,
        **kwargs,
    ):
        prompt = f"<image>\n\n{chat_history}Question: {question}\n\nAnswer:"
        answer = self.generate(
            image_embeds,
            prompt,
            tokenizer=tokenizer,
            max_new_tokens=512,
            **kwargs,
        )[0]
        cleaned_answer = answer.strip()

        # Use the result_queue to pass the result if it is provided
        if result_queue:
            result_queue.put(cleaned_answer)
        else:
            return cleaned_answer

    def batch_answer(
        self,
        images,
        prompts,
        tokenizer,
        **kwargs,
    ):
        image_embeds = self.encode_image(images)

        templated_prompts = [
            f"<image>\n\nQuestion: {prompt}\n\nAnswer:" for prompt in prompts
        ]
        prompt_embs = [
            self.input_embeds(prompt, image_embed.unsqueeze(0), tokenizer)[0]
            for prompt, image_embed in zip(templated_prompts, image_embeds)
        ]

        bos_emb = prompt_embs[0][0]
        max_len = max([p.shape[0] for p in prompt_embs])

        inputs_embeds = torch.cat(
            [
                torch.cat([bos_emb.repeat(max_len - p.shape[0], 1), p]).unsqueeze(0)
                for p in prompt_embs
            ],
            dim=0,
        )
        attention_mask = torch.cat(
            [
                torch.cat(
                    [
                        torch.zeros(
                            1,
                            max_len - p.shape[0],
                            device=self.device,
                            dtype=torch.long,
                        ),
                        torch.ones(1, p.shape[0], device=self.device, dtype=torch.long),
                    ],
                    dim=1,
                )
                for p in prompt_embs
            ],
            dim=0,
        )

        generate_config = {
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.bos_token_id,
            "max_new_tokens": 512,
            **kwargs,
        }

        with torch.no_grad():
            output_ids = self.text_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_config,
            )

        return [
            x.strip()
            for x in tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        ]
