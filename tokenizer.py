import pickle

class Tokenizer:
    """A simple tokenizer class based on Byte pair encoding and utf-8
    """
    def __init__(self, name, vocab_size = 300):
        self.name = name
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def train(self, dataset_path, save_path):
        """Trains the tokenizer on a given dataset

        Args:
            dataset_path (str): path to text file of dataset
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        ## Starts with encoding the text in utf-8 and then mapping them to a list of ints [0....255]
        encoded = text.encode('utf-8')
        tokens = list(map(int, encoded))
        
        new_tokens = tokens.copy()
        for token_number in range(256, self.vocab_size):
            common_pairs = self.get_common_pairs(new_tokens)
            pair = max(common_pairs, key=common_pairs.get)
            print(f"Merging {pair} into {token_number}")
            new_tokens = self.merge(new_tokens, pair, token_number)
            self.merges[pair] = token_number
        
        for pair, token_number in self.merges.items():
            self.vocab[token_number] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
        with open(save_path, 'wb') as file:
            pickle.dump(self.merges, file)
            
    def load(self, path='tokenizer.pkl'):
        with open(path, 'rb') as file:
            self.merges = pickle.load(file)
        for pair, token_number in self.merges.items():
            self.vocab[token_number] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
    def get_common_pairs(self, tokens):
        """Finds the pairs in a given list of tokens

        Args:
            tokens (list): List of token ids

        Returns:
            dict: dictionary with keys as pairs(tuple) of bytes and values as the frequency
        """
        common_pairs = {}
        for pair1, pair2 in zip(tokens, tokens[1:]):
            pair = (pair1, pair2)
            common_pairs[pair] = common_pairs.get(pair, 0)+1
        return common_pairs
    
    def merge(self, tokens, pair, new_token):
        """Merges tokens in text based on a given pair and new token number

        Args:
            tokens (list): List of token ids
            pair (tuple): pair of token ids
            new_token (int): New token number

        Returns:
            list: Updated list of token ids
        """
        tokens_cp = []
        i = 0
        while i < len(tokens):
            if i<len(tokens)-1 and pair == (tokens[i], tokens[i+1]):
                tokens_cp.append(new_token)
                i+=2
            else:
                tokens_cp.append(tokens[i])
                i+=1
        return tokens_cp
    
    def encode(self, text):
        """Encodes a given text into token ids

        Args:
            text (str): Input text

        Returns:
            list: List of token ids
        """
        encoded = text.encode('utf-8')
        tokens = list(map(int, encoded))
        while len(tokens)>=2:
            common_pairs = self.get_common_pairs(tokens)
            # Gives the pair corresponding to the minimum token number (means the new token number with highest occurings in the training set) 
            # from the trained merges based on the common pairs in the given text
            pair = min(common_pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair in self.merges:
                tokens = self.merge(tokens, pair, self.merges[pair])
            else:
                break
        return tokens     
    
    def decode(self, tokens):
        """Decodes a list of token ids into a string

        Args:
            tokens (list): List of token ids

        Returns:
            str: Output string
        """
        tokens = b"".join(self.vocab[token] for token in tokens)
        text = tokens.decode("utf-8", errors="replace")
        return text
    

if __name__ == "__main__":
    dataset_path = '/root/data/rrr/usr/gpt-2/data/input.txt'
    tokenizer = Tokenizer("shakespeare-gpt")
    tokenizer.train(dataset_path)
    inp = "Hellow thou!!"
    print(f"Input is : {inp}")
    test_tokens = tokenizer.encode(inp)
    print(f"Tokens are : {test_tokens}")
    decoded_text = tokenizer.decode(test_tokens)
    print(f"Decoded text is : {decoded_text}")
    
    
