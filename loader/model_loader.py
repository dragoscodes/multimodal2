import torch
from transformers import BitsAndBytesConfig , CLIPVisionModel, CLIPImageProcessor, LlamaForCausalLM, AutoTokenizer , LlamaTokenizer , AutoModelForCausalLM
  
quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

def load_vision_model(pretrained_model, device = "cpu", cache_dir = None ):
    vision_model = CLIPVisionModel.from_pretrained(pretrained_model, cache_dir=cache_dir, device=device)
    image_processor = CLIPImageProcessor.from_pretrained(pretrained_model,  cache_dir=cache_dir)
    return vision_model , image_processor;

def load_llm(pretrained_model, device = "cpu", cache_dir = None , quantization_config = None , attn_implementation = None):
    if attn_implementation is None:
        llm = AutoModelForCausalLM.from_pretrained(pretrained_model,  cache_dir=cache_dir, quantization_config = quantization_config).to(device)
    else:
        llm = LlamaForCausalLM.from_pretrained(pretrained_model,  cache_dir=cache_dir, quantization_config = quantization_config, attn_implementation = attn_implementation)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, cache_dir=cache_dir )
    return llm, tokenizer

def load_model_and_tokenizer(
    model_path, tokenizer_path=None, device="cuda", access_token=None, **kwargs
):
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            **kwargs,
        )
        .to(device)
        .eval()
    )

    model.requires_grad_(False)

    if "Llama-3" in model_path or "llama-3" in model_path:
        tokenizer = Tokenizer("llama_tokenizer.model")
    else:
        tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False,
            token=access_token,
        )
        tokenizer_path = tokenizer_path.lower()
        # TODO: no idea if this is true or not
        if "llama-3" in tokenizer_path:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

