import torch
from transformers import BitsAndBytesConfig , CLIPVisionModel, CLIPImageProcessor, LlamaForCausalLM, AutoTokenizer , LlamaTokenizer , AutoModelForCausalLM
  
quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

def load_vision_model(pretrained_model, device = "cpu", cache_dir = None ):
    vision_model = CLIPVisionModel.from_pretrained(pretrained_model, cache_dir=cache_dir).to(device)
    vision_model.share_memory()
    image_processor = CLIPImageProcessor.from_pretrained(pretrained_model,  cache_dir=cache_dir)
    return vision_model , image_processor;

def load_llm(pretrained_model, device = "cpu", cache_dir = None , quantization_config = None):
    llm = AutoModelForCausalLM.from_pretrained(pretrained_model,  cache_dir=cache_dir, quantization_config = None)
    llm.share_memory()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, cache_dir=cache_dir )
    return llm, tokenizer

def load_model_and_tokenizer(
    model_path, tokenizer_path=None, device="cuda", dtype=torch.bfloat16, access_token=None, **kwargs
):
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            token=access_token,
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