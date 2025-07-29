from collections import Counter
import torch
from collections import defaultdict
import heapq


def train_bpe(input_vocab, pre_tokens, num_merges):
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
    # print(f"Initial heap: {heap}")
    while len(merges) < num_merges and heap:
        # first find the most frequent pair
        negative_count, _, pair_to_be_merged = heapq.heappop(heap)
        if pair_to_be_merged not in successive_pair_count or successive_pair_count[pair_to_be_merged] != -negative_count:
            continue

        merges.append(pair_to_be_merged)
        merged_token = pair_to_be_merged[0] + pair_to_be_merged[1]
        vocab.append(merged_token)
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
        

        print(f"after merge {len(merges)}: {successive_pair_count}")
        print(f"{pre_tokens}")

        for pair in pairs_to_update:
            if pair in successive_pair_count and successive_pair_count[pair] > 0:
                heapq.heappush(heap, (-successive_pair_count[pair], reverse_bytes(pair), pair))
                
    return vocab, merges            


def reverse_bytes(byte_tuple):
    """Convert each bytes object to tuple of ALL its negative byte values"""
    return tuple(tuple(-b for b in bytes_obj) for bytes_obj in byte_tuple)

def merge(num, input_vocab, pre_tokens):
    """

    Args:
        num(int): number of merges conducted in the training
        input_vocab: the current states of the vocab
        pre_tokens: the resulting dict of pre-tokenization, consists of pretokens that consists of existing vocabs
    """

    if num == 0:
        return
    successive_pair_count = {}
    for k, v in pre_tokens.items():
        # k is a tuple of bytes
        for i in range(0, len(k) - 1):
            if (k[i], k[i + 1]) not in successive_pair_count:
                successive_pair_count[(k[i], k[i + 1])] = v
            else:
                successive_pair_count[(k[i], k[i + 1])] += v
    # update both the input_vocab and pre_tokens
    max_occurances = max(successive_pair_count.values())

    # I originally wants to first create a empty tuple acting as
    # a container, so that I can keep replacing the lex max
    # tuples

    potential_merges = [pair for pair, occurances in successive_pair_count.items() if occurances == max_occurances]
    merge_token = max(potential_merges)

    # update input_vocab
    new_token = merge_token[0] + merge_token[1]
    input_vocab.append(new_token)

    # update pre_tokens
    new_pre_tokens = {}
    for k, v in pre_tokens.items():
        new_k = []
        # for i in range(0, len(k) - 1):
        i = 0
        while i + 1 < len(k):
            if (k[i] + k[i + 1]) == new_token:
                new_k.append(new_token)
                i = i + 2
            else:
                new_k.append(k[i])
                i += 1
            if i+1 == len(k):
                new_k.append(k[i])
        new_k = tuple(elem for elem in new_k)
        new_pre_tokens[new_k] = v

    print(input_vocab[-1])
    print(new_pre_tokens)
    merge(num - 1, input_vocab, new_pre_tokens)

# print(vocab)
# print(d)
# print("--" * 20)

# merge(6, vocab, d)

# print(d)


if __name__ == "__main__":
    corpus = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""


    vocab = []
    vocab.append("<|endoftext|>".encode("utf-8"))
    for i in range(256):
        vocab.append(bytes([i]))
    

    # pre-tokenization

    pre_tokens = corpus.split()
    pre_token_count = Counter(pre_tokens)
    d = {}
    for token, count in pre_token_count.items():
        # convert a string: token into tuple of bytes
        byte_tuple = tuple(bytes([byte]) for byte in token.encode("utf-8"))
        d[byte_tuple] = count
    
    vocab, merges = train_bpe(vocab, d, 6)
    print("Final vocab:", vocab[256:])
    print("Merges:", merges)