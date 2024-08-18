from configparser import ConfigParser
import logging 
import os
import torch

class ConfigFileparser:
    """
    Parses configfile and stores them in attributes
    """
    def __init__(self, configfile = "config.ini"):
        parser = ConfigParser()
        parser.read(configfile)
        self.cwd = os.getcwd()
    
        self.context_size = int(parser.get('GPT','CONTEXT_SIZE',fallback=128))
        self.vocab_size = int(parser.get('GPT','VOCAB_SIZE',fallback=300))
        self.embedding_dim = int(parser.get('GPT','EMBEDDING_DIM',fallback=64))
        self.num_heads = int(parser.get('GPT','NUM_HEADS',fallback=4))
        self.num_blocks = int(parser.get('GPT','NUM_TRANSFORMER_BLOCK',fallback=4))
        
        self.dataset_path = parser.get('TRAIN','DATASET_PATH')
        self.dataset_path = os.path.join(self.cwd, self.dataset_path)
        
        self.saved_weights_path = parser.get('TRAIN','SAVED_WEIGHTS_PATH')
        self.saved_weights_path = os.path.join(self.cwd, self.saved_weights_path)
        
        self.batch_size = int(parser.get('TRAIN','BATCH_SIZE',fallback=32))
        self.lr = float(parser.get('TRAIN','LEARNING_RATE',fallback=1e-3))
        self.epochs = int(parser.get('TRAIN','EPOCHS',fallback=200))
        self.load_weights = parser.getboolean('TRAIN','LOAD_WEIGHTS',fallback=False)
        
        self.tokenizer_save_path = parser.get('TOKENIZER','TOKENIZER_SAVE_PATH')
        self.tokenizer_save_path = os.path.join(self.cwd, self.tokenizer_save_path)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

def get_config(configfile):
    return ConfigFileparser(configfile)
