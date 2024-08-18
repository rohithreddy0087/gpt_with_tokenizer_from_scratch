
from tokenizer import Tokenizer
from model import GPT
from config_parser import get_config

import torch
import torch.nn.functional as F

torch.manual_seed(1337)

def get_tokenizer(save_path):
    tokenizer = Tokenizer("shakespeare-gpt")
    tokenizer.load(save_path)
    print("Loaded Tokenizer")
    return tokenizer

if __name__ == '__main__':
    
    config_path = '/root/data/rrr/usr/gpt/config.ini'
    config = get_config(config_path)
    
    tokenizer = get_tokenizer(config.tokenizer_save_path)
    
    model = GPT(vocab_size=config.vocab_size, 
                context_size=config.context_size, 
                embedding_dim=config.embedding_dim, 
                num_heads=config.num_heads, 
                num_blocks=config.num_blocks,
                device=config.device).to(config.device)    
    model.load_state_dict(torch.load(config.saved_weights_path))
    model.eval()
    
    context = torch.tensor([[0]], device=config.device)
    try:
        while True:
            with torch.no_grad():
                inp = context[:,-config.context_size:].to(config.device)    
                logits = model(inp)
                logits = logits[:, -1, :] 
                probs = F.softmax(logits, dim=-1) 
                next_token = torch.multinomial(probs, num_samples=1) 
                context = torch.cat((context, next_token), dim=1)
                generated_tokens = tokenizer.decode(next_token[0].tolist())
                print(generated_tokens, end='', flush=True)
    except KeyboardInterrupt:
        print("\n--------------------------------")
        print("\nGPT generation ended by user.")