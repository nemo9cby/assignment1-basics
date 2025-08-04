from typing import Dict, List, Tuple
from cs336_basics.pretokenization_example import find_chunk_boundaries
# from cs336_basics.pretokenization_impl import pretokenize_sequential
from multiprocessing import Pool
from collections import Counter
import regex as re
from collections import defaultdict
import heapq
import json

def train_bpe(input_path, vocab_size=1000, special_tokens=None):
    vocab = {}
    vocab_id = 0
    
    # Add special tokens first
    for special_token in (special_tokens or []):
        vocab[vocab_id] = special_token.encode("utf-8")
        vocab_id += 1
    
    # Add all byte tokens (0-255)
    for i in range(256):
        vocab[vocab_id] = bytes([i])
        vocab_id += 1
    
    merges = []
    params = []
    num_processes = 8  # Adjust as needed
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            param = (f.name, start, end, special_tokens)
            params.append(param)
            
    with Pool(num_processes) as p:
        results = p.map(process_chunk, params) # if i dont add results, it will prompt errors
    pre_token_counter = Counter()
    for counter in results:
        pre_token_counter.update(counter)
    # pre_token_counter = process_chunk(params[0])
    # for start, end in zip(boundaries[:-1], boundaries[1:]):
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #         # Run pre-tokenization on your chunk and store the counts for each pre-token
    #         pre_tokens = chunk.split()
    #         pre_token_counter = Counter(pre_tokens)
    
            # return pre_token_count
    # Combine all counters from chunks
    # total_counts = Counter()
    # for counter in results:
    #     total_counts.update(counter)
    
    # TODO: Implement actual BPE training algorithm here
    # For now, just return the basic vocab with no merges
    
    vocab, merges = merge(vocab, pre_token_counter, vocab_size - len(vocab))
    # while len(vocab) < vocab_size: # there might be another condition where the chunk is too small
        # merge(vocab, total_counts)
    #    break
    
    return vocab, merges

def merge(input_vocab, pre_tokens, num_merges):
    # the key of pre_tokens is a tuple of "bytes", it is a class.
    vocab = input_vocab.copy()
    merges = []
    if num_merges == 0 or len(input_vocab) > 1000:
        return vocab, merges

    print("Num of pretokens:", len(pre_tokens))

    # step 0: create a dict to store the mapping between successive pairs and the pre-tokens
    
    pair_occurrences = defaultdict(set)
    heap = []
    # step 1: Find the most frequent pair of consecutive tokens, this only needs to be done once
    successive_pair_count = defaultdict(int)
    for k, v in pre_tokens.items():
        # k is a tuple of bytes
        for i in range(0, len(k) - 1):
            if (k[i], k[i + 1]) not in successive_pair_count:
                successive_pair_count[(k[i], k[i + 1])] = v
            else:
                successive_pair_count[(k[i], k[i + 1])] += v
            pair_occurrences[(k[i], k[i + 1])].add(k)
    
    # here we also track each pair's occurances in the pair_occurrences dict

    for pair, count in successive_pair_count.items():
        heapq.heappush(heap, (-count, reverse_bytes(pair),pair))  # Use negative count for max-heap behavior
    
    
    
    # # print(f"Initial heap: {heap}")
    
    
    while len(merges) < num_merges and heap:
        # first find the most frequent pair
        # negative_count, _, pair_to_be_merged = heapq.heappop(heap)
        pair_to_be_merged_iter = max(successive_pair_count, key=lambda pair: (successive_pair_count[pair], pair))

        # print(len(merges), "merges found so far")
        
        while heap:
            neg_count, neg_pair, pair = heapq.heappop(heap)
            current_count = successive_pair_count.get(pair, 0)
            
            if current_count == -neg_count and current_count > 0:
                pair_to_be_merged = pair
                break
        else:
            break
        
        # if pair_to_be_merged not in successive_pair_count or \
        #     successive_pair_count[pair_to_be_merged] != -negative_count:
        #     continue
        # assert pair_to_be_merged_iter == pair_to_be_merged, f"Expected {pair_to_be_merged_iter}, got {pair_to_be_merged}"
        if pair_to_be_merged_iter != pair_to_be_merged:
            print(f"Warning: Expected {pair_to_be_merged_iter}, got {pair_to_be_merged}")
            
        pair_to_be_merged = pair_to_be_merged_iter

        

        merges.append(pair_to_be_merged)
        merged_token = pair_to_be_merged[0] + pair_to_be_merged[1]
        # vocab.append(merged_token)
        vocab[len(vocab)] = merged_token
       # heapq.heappush(heap, (-negative_count, reverse_bytes(merged_token), merged_token))
        pairs_to_update = set()

        # because of the merge, there are new pairs can be formed, and need to be added to the heap,
        affected_tokens = list(pair_occurrences[pair_to_be_merged])
        for affected_token in affected_tokens:
            # each affected token is a tuple of bytes, and it contains the successive pair
            if affected_token not in pre_tokens:
                continue
            i = 0
            new_token_list = []
            
            while i < len(affected_token):
                # Check if we can merge at this position
                if (i < len(affected_token) - 1 and 
                    (affected_token[i], affected_token[i + 1]) == pair_to_be_merged):
                    new_token_list.append(merged_token)
                    i += 2  # Skip both parts of the merged pair
                else:
                    new_token_list.append(affected_token[i])
                    i += 1
            new_token = tuple(new_token_list)
            # affected_token is the old token, it should not appear in the pre_tokens anymore
            
            count = pre_tokens[affected_token]

            # problematic
            for i in range(len(affected_token) - 1):
                pair = (affected_token[i], affected_token[i + 1]) # we might be able to optimize this with offsets
                successive_pair_count[pair] -= count
                pairs_to_update.add(pair)
                if successive_pair_count[pair] == 0:
                    del successive_pair_count[pair]
                pair_occurrences[pair].discard(affected_token)
            
            # Add counts for new pairs
            
            for i in range(len(new_token) - 1):
                pair = (new_token[i], new_token[i + 1])
                pairs_to_update.add(pair)
                successive_pair_count[pair] += count
                pair_occurrences[pair].add(new_token)
                
                # problems here
            
            # Update pre_tokens
            del pre_tokens[affected_token]
            pre_tokens[new_token] = count
        # end merging affected tokens
        

        # print(f"after merge {len(merges)}: {successive_pair_count}")
        # print(f"{pre_tokens}")

        if pair_to_be_merged in successive_pair_count:
            del successive_pair_count[pair_to_be_merged]
        if pair_to_be_merged in pair_occurrences:
            del pair_occurrences[pair_to_be_merged]

        for pair in pairs_to_update:
            if pair in successive_pair_count and successive_pair_count[pair] > 0:
                heapq.heappush(heap, (-successive_pair_count[pair], reverse_bytes(pair), pair))
                
    return vocab, merges

def reverse_bytes(byte_tuple):
    """Convert each bytes object to tuple of ALL its negative byte values"""
    # return tuple(tuple(-b for b in bytes_obj) for bytes_obj in byte_tuple)
    def negate_bytes(b):
        # Create a tuple that sorts in reverse lexicographic order
        # Pad with positive values to handle length differences
        max_len = 1000  # Longer than any token we'll see
        result = []
        for byte_val in b:
            result.append(-byte_val)
        # Pad with positive values so shorter strings sort AFTER longer ones
        result.extend([1] * (max_len - len(b)))
        return tuple(result)
    
    return tuple(negate_bytes(byte_obj) for byte_obj in byte_tuple)

def process_chunk(chunk_info) -> Counter: # need to change the parameter to a tuple
    """
    Process the chunk for pre-tokenization.
    This function should implement the logic to count pre-tokens.
    """
    # Example implementation, replace with actual logic
    file_path, start, end, special_tokens = chunk_info
    with open(file_path, "rb") as f:
        f.seek(start)
        # Read the chunk from start to end
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # Remove special tokens and split the chunk
        if special_tokens:
            # Create regex pattern for splitting on special tokens
            # Escape special characters in tokens for regex
            escaped_tokens = [re.escape(token) for token in special_tokens]
            pattern = "|".join(escaped_tokens)
            # Split on special tokens to prevent merging across them
            segments = re.split(pattern, chunk)
        else:
            segments = [chunk]

        print(len(segments), "segments found in chunk")

        # print(segments[:5])
        
        # GPT-2 pretokenization pattern
        # This pattern splits on whitespace and punctuation
        # gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # For Python's regex module, we need a simplified version
        # This pattern captures contractions, words, numbers, and other characters
        # pretokenize_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
        pretokenize_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_token_count = Counter()
        
        # Process each segment separately
        for segment in segments:
            if segment:  # Skip empty segments
                # Apply pretokenization using regex
                pre_tokens = re.finditer(pretokenize_pattern, segment)
                # Update counter with pre-tokens from this segment
                for match in pre_tokens:
                    token = match.group()  # Extract the matched text
                    # token_bytes = token.encode('utf-8')  # Convert to bytes
                    byte_tuple = tuple(bytes([byte]) for byte in token.encode("utf-8"))
                    pre_token_count[byte_tuple] += 1

        return pre_token_count
    

def serialize_bpe_standard(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], 
                          vocab_path: str, merges_path: str):
    """
    Serialize vocab and merges in the standard BPE format used by 
    Hugging Face, OpenAI GPT-2, etc.
    
    Args:
        vocab: Dict mapping token IDs to token bytes
        merges: List of merge rules as (token1_bytes, token2_bytes) tuples
        vocab_path: Path to save vocab.json
        merges_path: Path to save merges.txt
    """
    
    # 1. Create vocab.json (token_string -> token_id)
    vocab_dict = {}
    for token_id, token_bytes in vocab.items():
        try:
            # Try to decode as UTF-8 first
            token_str = token_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Fall back to latin-1 for byte-level tokens
            token_str = token_bytes.decode('latin-1')
        vocab_dict[token_str] = token_id
    
    # Save vocab.json
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
    
    # 2. Create merges.txt (space-separated merge pairs)
    with open(merges_path, 'w', encoding='utf-8') as f:
        for merge1_bytes, merge2_bytes in merges:
            try:
                # Try UTF-8 first
                token1 = merge1_bytes.decode('utf-8')
                token2 = merge2_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # Fall back to latin-1
                token1 = merge1_bytes.decode('latin-1')
                token2 = merge2_bytes.decode('latin-1')
            
            f.write(f"{token1} {token2}\n")
    
    print(f"Saved vocab to {vocab_path} ({len(vocab_dict)} tokens)")
    print(f"Saved merges to {merges_path} ({len(merges)} merge rules)")



def save_trained_tokenizer(vocab, merges, output_dir="./tokenizer_output"):
    """Save your trained BPE tokenizer in standard format"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    vocab_path = os.path.join(output_dir, "vocab.json")
    merges_path = os.path.join(output_dir, "merges.txt")
    
    serialize_bpe_standard(vocab, merges, vocab_path, merges_path)
    
    # Inspect the results
    # print("\n--- Vocabulary Inspection ---")
    # inspect_vocab(vocab_path)
    
    # print("\n--- Merges Inspection ---")
    # inspect_merges(merges_path)
    
    return vocab_path, merges_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_bpe.py <input_file> [<vocab_size>]")
        sys.exit(1)

    input_file = sys.argv[1]
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    vocab, merges = train_bpe(input_path=input_file, vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
    save_trained_tokenizer(vocab, merges, output_dir="./tokenizer_output")
    
    