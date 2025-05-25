# Resources

This directory stores data and configuration files required by the transformer model implementations in this repository.

## Contents

-   **`config/`**: Contains model configuration files.
    -   **`gpt2/config.json`**: An example configuration file for a GPT-2 model, specifying parameters like embedding dimensions, number of layers, heads, vocabulary size, etc. This is typically used by the model loading logic to set up the correct architecture.

-   **`tokenizer_data/`**: Contains files necessary for the tokenizer.
    -   **`gpt2/`**: Specific tokenizer files for GPT-2 models.
        -   **`gpt2-vocab.json`**: The vocabulary file, mapping tokens (strings) to their corresponding IDs (integers).
        -   **`merges.txt`**: The BPE (Byte Pair Encoding) merges file, defining the merge rules used by the tokenizer to combine sub-word units.
        -   **`sample_token_ids.json`**: A sample file containing token IDs, likely used for testing the tokenizer or model.

These resources are essential for both loading pre-trained models and ensuring the tokenizer behaves consistently with the model it's being used for.
