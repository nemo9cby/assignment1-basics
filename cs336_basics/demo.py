from collections import Counter
import torch

corpus = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""


vocab = []
for i in range(256):
    vocab.append(bytes([i]))
vocab.append("<|endoftext|>")

# pre-tokenization

pre_tokens = corpus.split()
pre_token_count = Counter(pre_tokens)
d = {}
for token, count in pre_token_count.items():
    # convert a string: token into tuple of bytes
    byte_tuple = tuple(bytes([byte]) for byte in token.encode("utf-8"))
    d[byte_tuple] = count


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

print(vocab)
print(d)
print("--" * 20)

merge(6, vocab, d)

# print(d)
