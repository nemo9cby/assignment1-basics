from typing import Iterable, Iterator
import json
import regex as re

def create_byte_to_unicode():
    """
    Create a mapping from every byte (0-255) to a unique Unicode character.
    This is the standard mapping used by GPT-2 and HuggingFace tokenizers.
    """
    # Start with printable characters (excluding some problematic ones)
    bs = list(range(ord("!"), ord("~")+1)) + \
        list(range(ord("¡"), ord("¬")+1)) + \
        list(range(ord("®"), ord("ÿ")+1))
    
    cs = bs[:]
    n = 0
    
    # Map remaining bytes to unused Unicode code points
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    
    return dict(zip(bs, [chr(c) for c in cs]))

class Tokenizer:

    BYTE_TO_UNICODE = create_byte_to_unicode()
    UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}
    
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab # vocab is id to str
        self.merges = merges
        self.special_tokens = special_tokens
        
        
        self.vocab_bytes_to_id = {}
        # self.vocab_str_to_id = {}


        for token_id, token_bytes in self.vocab.items():
            self.vocab_bytes_to_id[token_bytes] = token_id
            
        

        self.special_tokens_dict = {}
        if special_tokens:
            for special_token in special_tokens:
                # Find the special token in the vocabulary
                # Special tokens are usually stored as-is in vocab.json
                special_token_bytes = special_token.encode('utf-8')
                if special_token_bytes in self.vocab_bytes_to_id:
                    self.special_tokens_dict[special_token] = self.vocab_bytes_to_id[special_token_bytes]
                else:
                    raise ValueError(f"Special token '{special_token}' not found in vocabulary")
    

        self.merge_priorities = {merge: i for i, merge in enumerate(merges)}


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        
        byte_to_unicode = cls.BYTE_TO_UNICODE
        unicode_to_byte = cls.UNICODE_TO_BYTE
        
        # 1. Load vocab.json
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
        
        vocab = {}
        
        for token_str, token_id in vocab_json.items():
            # Convert unicode string back to bytes
            token_bytes = bytes([unicode_to_byte[c] for c in token_str])
            vocab[token_id] = token_bytes

        # 2. Load merges.txt
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and version header
                if not line or line.startswith('#'):
                    continue
                
                token1_str, token2_str = line.split(' ')
                
                # Convert unicode strings back to bytes
                token1_bytes = bytes([unicode_to_byte[c] for c in token1_str])
                token2_bytes = bytes([unicode_to_byte[c] for c in token2_str])
                
                merges.append((token1_bytes, token2_bytes))
        
        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
        # Normal BPE encoding
            return self._encode_ordinary(text)
    
        # Split text by special tokens while preserving them
        segments = self._split_on_special_tokens(text)
    
        result = []
        for segment in segments:
            if segment in self.special_tokens:
                # Direct mapping for special tokens
                # Find out the special tokens's vocab
                result.append(self.special_tokens_dict[segment])
            else:
                # Apply BPE to ordinary text
                result.extend(self._encode_ordinary(segment))
        
        return result

    def _encode_ordinary(self, segment):
        # encode all tokens other than special tokens
        # first pre-tokenize
        pretokenize_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = re.finditer(pretokenize_pattern, segment)
                # Update counter with pre-tokens from this segment
        result_ids = []
        for match in pre_tokens:
            token = match.group()  # Extract the matched text
            token_bytes = token.encode('utf-8')  # Convert to bytes
            token_bytes = [bytes([b]) for b in token_bytes]
            # tokens = [self.BYTE_TO_UNICODE[b] for b in token_bytes]
        
            # Apply BPE merges
            token_bytes_list = self._apply_bpe_merges(token_bytes)
            
            # Convert final tokens to IDs
            for merged_bytes in token_bytes_list:
                # print(token_str)
                result_ids.append(self.vocab_bytes_to_id[merged_bytes])

                # Convert back to bytes then look up ID
                # token_bytes = bytes([self.UNICODE_TO_BYTE[c] for c in token_str])
                # if token_bytes in self.bytes_to_id:
                #     result_ids.append(self.bytes_to_id[token_bytes])
        
        return result_ids
    
    def _apply_bpe_merges(self, tokens: list[bytes]) -> list[bytes]:
        """Apply merges to a list of string tokens - OPTIMIZED VERSION."""
        while len(tokens) > 1:
            # Find the pair with the lowest merge priority (earliest in merges list)
            best_pair = None
            best_priority = float('inf')
            best_position = -1
            
            # Check all adjacent pairs
            for i in range(len(tokens) - 1):
                pair = (bytes(tokens[i]), bytes(tokens[i + 1]))
                if pair in self.merge_priorities:
                    priority = self.merge_priorities[pair]  # O(1) lookup!
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair
                        best_position = i
            
            # If no more merges can be applied, we're done
            if best_pair is None:
                break
            
            # Apply the best merge
            first, second = best_pair
            merged = first + second
            tokens = tokens[:best_position] + [merged] + tokens[best_position + 2:]
        
        return tokens    

    def _split_on_special_tokens(self, text: str) -> list[str]:
        """Split text on special tokens while preserving them."""
        if not self.special_tokens:
            return [text]
        
        # Create regex pattern for all special tokens
        # import re

        sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)

        pattern = '(' + '|'.join(re.escape(token) for token in sorted_tokens) + ')'
        
        # Split while keeping the separators
        segments = re.split(pattern, text)
        
        # Filter out empty strings
        return [s for s in segments if s]


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode multiple texts, yielding token IDs one by one.
        Typically adds <endoftext> tokens between texts.
        """
        for i, text in enumerate(iterable):
            # Encode the current text
            token_ids = self.encode(text)
            
            # Yield all token IDs from this text
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        all_bytes = []
        
        for token_id in ids:
            if token_id in self.vocab:
                # vocab[token_id] gives us the actual bytes
                token_bytes = self.vocab[token_id]
                all_bytes.extend(token_bytes)
            else:
                # Handle unknown token (optional)
                # You might want to use a special <unk> token or skip
                print(f"Warning: Unknown token ID {token_id}")
        
        # Convert bytes to string
        # Use 'replace' to handle any invalid UTF-8 sequences gracefully
        text = bytes(all_bytes).decode('utf-8', errors='replace')
        
        return text



if __name__ == "__main__":
    # list_of_special_tokens = [''] 
    t = Tokenizer.from_files(vocab_filepath="/Users/nemo/Projects/assignment1-basics/tokenizer_output/tiny_head_1000_escape_space/vocab.json",
                             merges_filepath="/Users/nemo/Projects/assignment1-basics/tokenizer_output/tiny_head_1000_escape_space/merges.txt",
                             special_tokens=['<|endoftext|>'])
    
    test_str = "s"
    print(t.encode(test_str))
