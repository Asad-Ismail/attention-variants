import torch
import time
from tokenizer import CharacterTokenizer
from causal_attention import transformerDecoder
from utils import set_seed


def generate_with_cache(model, prompt_tokens, num_tokens=50):

    x = torch.tensor([prompt_tokens]).to(model.device)  # Shape: [1, prompt_length]
    # Process the entire prompt and get initial output and cache
    start_time = time.time()
    output, kv_cache = model(x, return_cache=True)
    
    # Generate tokens one by one
    generated_tokens = []
    for i in range(num_tokens):
        # Get next token prediction
        next_token_logits = output[:, -1, :]  # Take logits of the last token
        # Greedy decoding
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)  # Shape: [batch_size, 1]
        generated_tokens.append(next_token.item())
        
        # Process only new token with the cache
        output, kv_cache = model(next_token, cache=kv_cache, return_cache=True)
    
    end_time = time.time()
    
    return {
        "tokens": prompt_tokens + generated_tokens,
        "time_taken": end_time - start_time
    }


def generate_without_cache(model, prompt_tokens, num_tokens=50):
    # Start with the prompt tokens
    all_tokens=prompt_tokens
    x = torch.tensor([all_tokens]).to(model.device)
    
    start_time = time.time()
    
    # Generate tokens one by one
    for i in range(num_tokens):
        # Process the entire sequence so far
        output = model(x, return_cache=False)
        
        # Get next token prediction
        next_token_logits = output[:, -1, :]  # Take logits of the last token
        # Greedy decoding
        next_token = torch.argmax(next_token_logits, dim=-1).item()
        
        # Append the new token to our sequence
        all_tokens.append(next_token)
        
        # Create a new input tensor with the updated sequence
        x = torch.tensor([all_tokens]).to(model.device)
    
    end_time = time.time()
    
    return {
        "tokens": all_tokens,
        "time_taken": end_time - start_time
    }


if __name__=="__main__": 
    set_seed()
    inp_txt = "Sky is "
    tokenizer = CharacterTokenizer()
    input_ids= tokenizer.encode(inp_txt)
    model=transformerDecoder()
    model.to("cpu")
    out_without_cache=generate_without_cache(model,input_ids.copy())
    out_with_cache=generate_with_cache(model,input_ids.copy())
    print(f"Output without cache")
    print(out_without_cache["tokens"])
    print(f"Output with cache")
    print(out_with_cache["tokens"])
    assert out_without_cache["tokens"]==out_with_cache["tokens"], "Output is not same when using and not using cache :("