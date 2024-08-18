
from tokenizer import Tokenizer
from model import GPT
from config_parser import get_config

import torch
import torch.nn.functional as F

torch.manual_seed(1337)

def get_data(config):
    with open(config.dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = Tokenizer("shakespeare-gpt")
    tokenizer.train(config.dataset_path, config.tokenizer_save_path)
    tokens = tokenizer.encode(text)
    return tokens

if __name__ == '__main__':
    config_path = '/root/data/rrr/usr/gpt/config.ini'
    config = get_config(config_path)
    
    tokens = get_data(config)

    num_batches = int(len(tokens)/(config.batch_size*(config.context_size+1)))

    tokens_tensor = torch.tensor(tokens[:num_batches*config.batch_size*(config.context_size+1)])
    tokens_tensor = tokens_tensor.view(-1, config.batch_size, config.context_size+1)
    inputs = tokens_tensor[:,:,:-1]
    labels = tokens_tensor[:,:,1:]

    model = GPT(vocab_size=config.vocab_size, 
                context_size=config.context_size, 
                embedding_dim=config.embedding_dim, 
                num_heads=config.num_heads, 
                num_blocks=config.num_blocks,
                device=config.device).to(config.device)
    
    if config.load_weights is not None:
        model.load_state_dict(torch.load(config.saved_weights_path))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    
    
    for epoch in range(config.epochs):
        total_loss = 0
        for i in range(num_batches):
            inp = inputs[i,:,:].to(config.device)
            targets = labels[i,:,:].to(config.device)
            
            with torch.autocast(device_type = config.device, dtype=torch.float16):
                logits = model(inp)
                
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss
        torch.save(model.state_dict(), config.saved_weights_path)
        print(f"{epoch}, Loss: {total_loss/num_batches}")
        
        