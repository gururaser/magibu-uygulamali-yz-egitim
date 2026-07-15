import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

"""Byte Pair Encoding (BPE) Tokenizer for Movie Titles.

This module provides tools to train a BPE tokenizer using Hugging Face's tokenizers 
library on preprocessed movie titles, and load it using a wrapper class BPETokenizer
that mimics character-level tokenizers with built-in preprocessing.
"""

class BPETokenizer:
    """A BPE Tokenizer wrapper.

    Wraps the trained Hugging Face Byte-Level BPE tokenizer to provide
    clean encode and decode interfaces, with automated preprocessing.

    Attributes:
        tokenizer (Tokenizer): The underlying Hugging Face Tokenizer instance.
        vocab_size (int): Size of the vocabulary.
        pad_token_id (int): Token ID representing padding.
        bos_token_id (int): Token ID representing beginning of sequence.
        eos_token_id (int): Token ID representing end of sequence.
    """

    def __init__(self, tokenizer_path: str):
        """Initializes BPETokenizer from a saved tokenizer configuration file.

        Args:
            tokenizer_path (str): Path to the saved tokenizer json configuration.
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.bos_token_id = self.tokenizer.token_to_id("<bos>")
        self.eos_token_id = self.tokenizer.token_to_id("<eos>")

    def preprocess(self, text: str) -> str:
        """Preprocesses the input text to match the tokenizer's training format.

        This includes standard lowercasing and whitespace normalization.

        Args:
            text (str): The raw input text.

        Returns:
            str: Preprocessed text.
        """
        # Match the preprocessing done on the training dataset (standard lowercasing)
        text = text.lower()
        text = " ".join(text.split())
        return text

    def encode(self, text: str) -> list[int]:
        """Preprocesses and encodes the text into token IDs.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[int]: List of token IDs.
        """
        cleaned_text = self.preprocess(text)
        return self.tokenizer.encode(cleaned_text).ids

    def decode(self, ids: list[int]) -> str:
        """Decodes the list of token IDs back into a string.

        Args:
            ids (list[int]): List of token IDs.

        Returns:
            str: The decoded text.
        """
        return self.tokenizer.decode(ids)

def train_bpe_tokenizer(input_file: str, output_json_path: str, vocab_size: int = 2000):
    """Trains a Byte-Level BPE tokenizer on the specified text file.

    Args:
        input_file (str): Path to the text file containing the training corpus.
        output_json_path (str): Path where the trained tokenizer should be saved.
        vocab_size (int, optional): Vocabulary size for the BPE model. Defaults to 2000.
    """
    print(f"Training BPE tokenizer on '{input_file}' with vocab_size={vocab_size}...")
    
    # We use ByteLevel pre-tokenizer so that it can handle any character gracefully
    # and match modern LLM tokenizers (like GPT, Llama, Qwen, etc.).
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]
    )
    
    tokenizer.train([input_file], trainer)
    
    tokenizer.save(output_json_path)
    print(f"Tokenizer saved to '{output_json_path}'")

def main():
    """Main function to run tokenizer training and print tokenization samples."""
    # Since train_tokenizer.py is now in the main les1/ folder, 
    # the data files are under the 'data' subfolder.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "data", "movie_titles.txt")
    output_path = os.path.join(current_dir, "data", "bpe_tokenizer.json")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at {input_file}. Please run download_data.py first.")
        
    train_bpe_tokenizer(input_file, output_path, vocab_size=2000)
    
    # Load and test
    bpe_tok = BPETokenizer(output_path)
    print(f"\nBPETokenizer successfully loaded! Vocab size: {bpe_tok.vocab_size}")
    
    # Let's test it with some sample movie titles
    samples = [
        "The Dark Knight",
        "Hababam Sınıfı",
        "Interstellar",
        "Pulp Fiction",
        "Inception"
    ]
    
    print("\n--- Tokenization Samples ---")
    for sample in samples:
        encoded = bpe_tok.encode(sample)
        decoded = bpe_tok.decode(encoded)
        # Convert tokens to their string representations for debugging/inspection
        tokens = [bpe_tok.tokenizer.id_to_token(i) for i in encoded]
        print(f"Original: '{sample}'")
        print(f"Tokens:   {tokens}")
        print(f"IDs:      {encoded}")
        print(f"Decoded:  '{decoded}'")
        print("-" * 30)

if __name__ == "__main__":
    main()
