#!/usr/bin/env python3
"""
Memory-efficient dataloader using np.memmap for large datasets.
Supports streaming tokenization and caching.
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Iterator
import json
from threading import Thread, Lock
from queue import Queue
import time


class MemmapDataLoader:
    """
    Memory-efficient dataloader that uses memory-mapped files for large datasets.

    Features:
    - Caches tokenized data to disk using np.memmap
    - Supports streaming tokenization (tokenize while training)
    - Memory-efficient: only loads what's needed into RAM
    - Thread-safe for parallel tokenization
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        cache_dir: str = "./data_cache",
        chunk_size: int = 1_000_000,  # Characters to tokenize at once
        dtype: np.dtype = np.int32,  # int32 supports vocab up to 2B tokens
    ):
        """
        Initialize the memory-mapped dataloader.

        Args:
            data_path: Path to the text data file
            tokenizer: Tokenizer instance with encode() method
            cache_dir: Directory to store cached memmap files
            chunk_size: Size of text chunks to tokenize at once (in characters)
            dtype: Data type for storing token IDs
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)
        self.chunk_size = chunk_size
        self.dtype = dtype

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache file names based on data file
        self.cache_name = f"{self.data_path.stem}_tokens"
        self.memmap_path = self.cache_dir / f"{self.cache_name}.npy"
        self.info_path = self.cache_dir / f"{self.cache_name}_info.json"

        # Load or create memory-mapped array
        self.tokens = None
        self.num_tokens = 0
        self._setup_memmap()

    def _setup_memmap(self):
        """Setup memory-mapped array, either loading existing or creating new."""
        if self.memmap_path.exists() and self.info_path.exists():
            # Load existing memmap
            print(f"Loading cached tokenized data from {self.memmap_path}")
            with open(self.info_path, 'r') as f:
                info = json.load(f)

            self.num_tokens = info['num_tokens']
            self.tokens = np.memmap(
                self.memmap_path,
                dtype=self.dtype,
                mode='r',  # Read-only for training
                shape=(self.num_tokens,)
            )
            print(f"  Loaded {self.num_tokens:,} tokens from cache")
        else:
            # Need to tokenize and create memmap
            print(f"No cache found. Tokenizing {self.data_path}...")
            self._tokenize_and_cache()

    def _tokenize_and_cache(self):
        """Tokenize the data file and save to memory-mapped array."""
        overall_start = time.time()

        # First pass: count total tokens needed
        print("  First pass: Counting tokens...")
        pass1_start = time.time()
        total_tokens = self._count_tokens()
        pass1_time = time.time() - pass1_start
        print(f"  Total tokens: {total_tokens:,} (took {pass1_time:.1f}s)")

        # Create memory-mapped array
        print(f"  Creating memory-mapped array at {self.memmap_path}")
        self.tokens = np.memmap(
            self.memmap_path,
            dtype=self.dtype,
            mode='w+',  # Write mode for creation
            shape=(total_tokens,)
        )

        # Second pass: tokenize and fill the array
        print("  Second pass: Tokenizing and saving...")
        pass2_start = time.time()
        self._fill_memmap(total_tokens)
        pass2_time = time.time() - pass2_start

        # Save metadata
        info = {
            'num_tokens': total_tokens,
            'data_path': str(self.data_path),
            'chunk_size': self.chunk_size,
            'dtype': str(self.dtype)
        }
        with open(self.info_path, 'w') as f:
            json.dump(info, f, indent=2)

        total_time = time.time() - overall_start
        file_size_mb = os.path.getsize(self.data_path) / (1024 * 1024)
        avg_speed = file_size_mb / total_time if total_time > 0 else 0
        print(f"  ✓ Tokenization complete. Cached to {self.memmap_path}")
        print(f"    Total time: {total_time:.1f}s (Pass 1: {pass1_time:.1f}s, Pass 2: {pass2_time:.1f}s)")
        print(f"    Average speed: {avg_speed:.1f} MB/s for {file_size_mb:.1f} MB file")

        # Reopen as read-only
        self.num_tokens = total_tokens
        del self.tokens
        self.tokens = np.memmap(
            self.memmap_path,
            dtype=self.dtype,
            mode='r',
            shape=(self.num_tokens,)
        )

    def _count_tokens(self) -> int:
        """Count total tokens in the dataset."""
        total_tokens = 0
        file_size = os.path.getsize(self.data_path)
        bytes_processed = 0
        start_time = time.time()
        last_print_time = start_time

        with open(self.data_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break

                bytes_processed += len(chunk.encode('utf-8'))

                # Tokenize chunk
                token_ids = self.tokenizer.encode(chunk)
                total_tokens += len(token_ids)

                # Show progress every 2 seconds or every 1M tokens
                current_time = time.time()
                if total_tokens % 1_000_000 == 0 or current_time - last_print_time >= 2.0:
                    elapsed = current_time - start_time
                    progress = bytes_processed / file_size
                    mb_processed = bytes_processed / (1024 * 1024)
                    mb_total = file_size / (1024 * 1024)

                    if elapsed > 0:
                        speed_mb_s = mb_processed / elapsed
                        tokens_per_sec = total_tokens / elapsed
                        eta_seconds = (mb_total - mb_processed) / speed_mb_s if speed_mb_s > 0 else 0
                        eta_min = eta_seconds / 60

                        print(f"    Counting: {mb_processed:.1f}/{mb_total:.1f} MB ({progress*100:.1f}%) "
                              f"| {total_tokens:,} tokens | {speed_mb_s:.1f} MB/s | "
                              f"{tokens_per_sec:.0f} tok/s | ETA: {eta_min:.1f} min")

                    last_print_time = current_time

        return total_tokens

    def _fill_memmap(self, total_tokens: int):
        """Fill the memory-mapped array with tokenized data."""
        position = 0
        file_size = os.path.getsize(self.data_path)
        bytes_processed = 0
        start_time = time.time()
        last_print_time = start_time

        with open(self.data_path, 'r', encoding='utf-8') as f:
            while position < total_tokens:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break

                bytes_processed += len(chunk.encode('utf-8'))

                # Tokenize chunk
                token_ids = self.tokenizer.encode(chunk)

                # Write to memmap
                end_position = position + len(token_ids)
                self.tokens[position:end_position] = token_ids
                position = end_position

                # Show progress every 2 seconds or every 1M tokens
                current_time = time.time()
                if position % 1_000_000 == 0 or current_time - last_print_time >= 2.0:
                    elapsed = current_time - start_time
                    progress = position / total_tokens
                    mb_processed = bytes_processed / (1024 * 1024)
                    mb_total = file_size / (1024 * 1024)

                    if elapsed > 0:
                        speed_mb_s = mb_processed / elapsed
                        tokens_per_sec = position / elapsed
                        remaining_tokens = total_tokens - position
                        eta_seconds = remaining_tokens / tokens_per_sec if tokens_per_sec > 0 else 0
                        eta_min = eta_seconds / 60

                        print(f"    Tokenizing: {position:,}/{total_tokens:,} tokens ({progress*100:.1f}%) "
                              f"| {mb_processed:.1f}/{mb_total:.1f} MB | {speed_mb_s:.1f} MB/s | "
                              f"{tokens_per_sec:.0f} tok/s | ETA: {eta_min:.1f} min")

                    last_print_time = current_time

        # Ensure the memmap is flushed to disk
        del self.tokens._mmap

    def get_batch(
        self,
        batch_size: int,
        context_length: int,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random batch of data.

        Args:
            batch_size: Number of sequences in batch
            context_length: Length of each sequence
            device: Device to place tensors on

        Returns:
            Tuple of (inputs, targets) tensors
        """
        if self.num_tokens < context_length + 1:
            raise ValueError(f"Not enough tokens ({self.num_tokens}) for context_length={context_length}")

        # Sample random starting positions
        max_start = self.num_tokens - context_length - 1
        start_indices = np.random.randint(0, max_start, size=batch_size)

        # Create batch tensors
        inputs = torch.zeros(batch_size, context_length, dtype=torch.long)
        targets = torch.zeros(batch_size, context_length, dtype=torch.long)

        # Fill batch
        for i, start_idx in enumerate(start_indices):
            # Get sequence from memmap (this only loads needed data into RAM)
            sequence = self.tokens[start_idx:start_idx + context_length + 1]
            inputs[i] = torch.from_numpy(sequence[:-1].astype(np.int64))
            targets[i] = torch.from_numpy(sequence[1:].astype(np.int64))

        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        return inputs, targets

    def __len__(self):
        """Return total number of tokens."""
        return self.num_tokens

    def __repr__(self):
        return f"MemmapDataLoader(tokens={self.num_tokens:,}, cache={self.memmap_path})"


class StreamingMemmapDataLoader(MemmapDataLoader):
    """
    Advanced version that tokenizes in background while training.
    Useful for extremely large datasets where you want to start training immediately.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        cache_dir: str = "./data_cache",
        chunk_size: int = 1_000_000,
        dtype: np.dtype = np.int32,
        prefetch_chunks: int = 10,  # Number of chunks to prefetch
    ):
        self.prefetch_chunks = prefetch_chunks
        self.tokenize_queue = Queue(maxsize=prefetch_chunks)
        self.write_position = 0
        self.write_lock = Lock()
        self.tokenization_done = False

        # Don't call parent __init__ yet
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)
        self.chunk_size = chunk_size
        self.dtype = dtype

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_name = f"{self.data_path.stem}_tokens"
        self.memmap_path = self.cache_dir / f"{self.cache_name}.npy"
        self.info_path = self.cache_dir / f"{self.cache_name}_info.json"

        # Check if already cached
        if self.memmap_path.exists() and self.info_path.exists():
            # Use regular loading
            super().__init__(data_path, tokenizer, cache_dir, chunk_size, dtype)
        else:
            # Start streaming tokenization
            self._start_streaming_tokenization()

    def _start_streaming_tokenization(self):
        """Start background tokenization thread."""
        print(f"Starting streaming tokenization of {self.data_path}...")

        # Estimate total size (we'll grow the memmap as needed)
        file_size = self.data_path.stat().st_size
        estimated_tokens = file_size // 4  # Rough estimate: 4 chars per token

        # Create initial memmap (we'll resize if needed)
        self.num_tokens = 0
        self.allocated_size = estimated_tokens
        self.tokens = np.memmap(
            self.memmap_path,
            dtype=self.dtype,
            mode='w+',
            shape=(self.allocated_size,)
        )

        # Start tokenization thread
        self.tokenize_thread = Thread(target=self._tokenize_in_background)
        self.tokenize_thread.daemon = True
        self.tokenize_thread.start()

        # Wait for some initial data
        print("  Waiting for initial chunks to be tokenized...")
        while self.num_tokens < self.chunk_size // 4 and not self.tokenization_done:
            time.sleep(0.1)
        print(f"  Ready to start training! ({self.num_tokens:,} tokens available)")

    def _tokenize_in_background(self):
        """Background thread that tokenizes data."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break

                # Tokenize chunk
                token_ids = self.tokenizer.encode(chunk)

                # Add to queue for writing
                self.tokenize_queue.put(token_ids)

                # Write queued chunks
                self._write_queued_chunks()

        # Process remaining chunks
        self._write_queued_chunks()

        # Mark as done
        self.tokenization_done = True

        # Save final metadata
        info = {
            'num_tokens': self.num_tokens,
            'data_path': str(self.data_path),
            'chunk_size': self.chunk_size,
            'dtype': str(self.dtype)
        }
        with open(self.info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"\n  ✓ Background tokenization complete: {self.num_tokens:,} total tokens")

    def _write_queued_chunks(self):
        """Write tokenized chunks from queue to memmap."""
        while not self.tokenize_queue.empty():
            token_ids = self.tokenize_queue.get()

            with self.write_lock:
                # Check if we need to resize
                if self.write_position + len(token_ids) > self.allocated_size:
                    self._resize_memmap()

                # Write to memmap
                self.tokens[self.write_position:self.write_position + len(token_ids)] = token_ids
                self.write_position += len(token_ids)
                self.num_tokens = self.write_position

                # Show progress
                if self.num_tokens % 1_000_000 == 0:
                    print(f"    Tokenized {self.num_tokens:,} tokens (streaming)...")

    def _resize_memmap(self):
        """Resize the memory-mapped array when it's full."""
        new_size = int(self.allocated_size * 1.5)
        print(f"    Resizing memmap: {self.allocated_size:,} → {new_size:,}")

        # Create new larger memmap
        new_tokens = np.memmap(
            self.memmap_path.with_suffix('.tmp'),
            dtype=self.dtype,
            mode='w+',
            shape=(new_size,)
        )

        # Copy existing data
        new_tokens[:self.allocated_size] = self.tokens[:]

        # Replace old with new
        del self.tokens
        os.rename(self.memmap_path.with_suffix('.tmp'), self.memmap_path)

        self.tokens = np.memmap(
            self.memmap_path,
            dtype=self.dtype,
            mode='r+',
            shape=(new_size,)
        )
        self.allocated_size = new_size

    def get_batch(
        self,
        batch_size: int,
        context_length: int,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch, but only from already-tokenized data."""
        with self.write_lock:
            available_tokens = self.num_tokens

        if available_tokens < context_length + 1:
            # Wait for more tokens
            while self.num_tokens < context_length + 1:
                if self.tokenization_done:
                    raise ValueError(f"Not enough tokens for training")
                time.sleep(0.1)

        # Only sample from available tokens
        max_start = min(available_tokens - context_length - 1, self.num_tokens - context_length - 1)

        if max_start <= 0:
            raise ValueError(f"Not enough tokenized data yet. Available: {available_tokens}")

        # Rest is same as parent class
        start_indices = np.random.randint(0, max_start, size=batch_size)

        inputs = torch.zeros(batch_size, context_length, dtype=torch.long)
        targets = torch.zeros(batch_size, context_length, dtype=torch.long)

        for i, start_idx in enumerate(start_indices):
            sequence = self.tokens[start_idx:start_idx + context_length + 1]
            inputs[i] = torch.from_numpy(sequence[:-1].astype(np.int64))
            targets[i] = torch.from_numpy(sequence[1:].astype(np.int64))

        return inputs.to(device), targets.to(device)