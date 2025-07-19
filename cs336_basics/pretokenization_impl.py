from collections import Counter
import os
from typing import BinaryIO
from pretokenization_example import pretokenize_sequential
import time

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

## Usage

def process_chunk(chunk_info) -> Counter: # need to change the parameter to a tuple
    """
    Process the chunk for pre-tokenization.
    This function should implement the logic to count pre-tokens.
    """
    # Example implementation, replace with actual logic
    file_path, start, end = chunk_info
    with open(file_path, "rb") as f:
        f.seek(start)
        # Read the chunk from start to end
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        pre_tokens = chunk.split()
        pre_token_count = Counter(pre_tokens)

        return pre_token_count

def combine_counters(counter_list: list[Counter]) -> Counter:
    """Combine multiple Counter objects into one."""
    combined = Counter()
    for counter in counter_list:
        combined.update(counter)
    return combined

from multiprocessing import Pool

def main(filepath: str, num_processes: int = 1) -> Counter:
    
    params = []
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            param = (f.name, start, end)
            params.append(param)
        
            # f.seek(start)
            # chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
    with Pool(num_processes) as p:
        results = p.map(process_chunk,params) # if i dont add results, it will prompt errors
    
    final_counter = combine_counters(results)

    print(final_counter)
    return final_counter
    

def test_serial_parallel(c_seria, c_parallel):
    """
    Test if the serial and parallel implementations yield the same results.
    """
    # check if all keys in c_seria are in c_parallel
    assert all(key in c_parallel for key in c_seria), "Serial counter has keys not in parallel counter!"
    # check if all keys in c_parallel are in c_seria
    assert all(key in c_seria for key in c_parallel), "Parallel counter has keys not in serial counter!"
    
    
    assert c_seria == c_parallel, "Serial and parallel counters do not match!"


if __name__ == "__main__":
    start_parallel = time.time()
    parallel_counter = main(filepath="./data/TinyStoriesV2-GPT4-valid.txt",num_processes=4)
    end_parallel = time.time()
    start_sequential = time.time()
    sequential_counter = pretokenize_sequential("./data/TinyStoriesV2-GPT4-valid.txt")
    end_sequential = time.time()
    print("sequential_counter:", sequential_counter)
    test_serial_parallel(sequential_counter, parallel_counter)
    
    print("Both implementations yield the same results.")
    print(f"Parallel processing took {end_parallel - start_parallel:.2f} seconds.")
    print(f"Sequential processing took {end_sequential - start_sequential:.2f} seconds.")
    print(f"Speedup: {((end_sequential - start_sequential) / (end_parallel - start_parallel)):.2f}x")