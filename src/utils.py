import os
import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization
import matplotlib.pyplot as plt
import torch

def load_and_vectorize_data():
    url = "https://www.manythings.org/anki/fra-eng.zip"
    # Create the directory safely
    os.makedirs("../data", exist_ok=True)
    zip_path = "../data/fra-eng.zip"
    txt_path = "../data/fra.txt"

    if not os.path.exists(txt_path):
        print("Downloading English-French dataset with browser headers...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("../data/")
        print("Download and extraction complete.")

    # --- 2. LOAD RAW TEXT ---
    raw_eng, raw_fra_in, raw_fra_out = [], [], []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split('\t')
            if len(parts) >= 2:
                eng = parts[0].lower().strip()
                fra = parts[1].lower().strip()
                
                raw_eng.append(eng)
                raw_fra_in.append(f"startseq {fra}")
                raw_fra_out.append(f"{fra} endseq")

    # Slice a generous subset (e.g., 20,000) so your RAM doesn't crash, 
    # while keeping sentences of any arbitrary length!
    num_samples = 20000 
    eng_sentences = raw_eng[:num_samples]
    fra_in_sentences = raw_fra_in[:num_samples]
    fra_out_sentences = raw_fra_out[:num_samples]

    # --- 3. DYNAMICALLY FIND MAX LENGTHS ---
    # Instead of guessing 10 or 12, let's find the true max of this subset
    max_src_len = max(len(s.split()) for s in eng_sentences)
    max_trg_len = max(len(s.split()) for s in fra_in_sentences)

    # Source (English) Vectorizer
    src_vectorizer = TextVectorization(
        max_tokens=None,
        standardize="lower_and_strip_punctuation",
        output_mode="int",
        output_sequence_length=max_src_len
    )
    src_vectorizer.adapt(eng_sentences)

    # Target (French) Vectorizer
    def custom_standardize(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase, r"[.?!,¿]", "")

    trg_vectorizer = TextVectorization(
        max_tokens=None,
        standardize=custom_standardize,
        output_mode="int",
        output_sequence_length=max_trg_len
    )
    trg_vectorizer.adapt(fra_in_sentences + fra_out_sentences)

    # Convert strings directly to integer tensors
    encoder_input = src_vectorizer(np.array(eng_sentences))
    decoder_input = trg_vectorizer(np.array(fra_in_sentences))
    decoder_target = trg_vectorizer(np.array(fra_out_sentences))

    src_vocab_size = len(src_vectorizer.get_vocabulary())
    trg_vocab_size = len(trg_vectorizer.get_vocabulary())

    print(f"Vectorization Ready. English Vocab: {src_vocab_size}, French Vocab: {trg_vocab_size}")
    print(f"Max Source Length: {max_src_len}, Max Target Length: {max_trg_len}")

    src_vectorizer.vocab_size = len(src_vectorizer.get_vocabulary())
    trg_vectorizer.vocab_size = len(trg_vectorizer.get_vocabulary())
    trg_vectorizer.word_to_index = {word: i for i, word in enumerate(trg_vectorizer.get_vocabulary())}
    
    # Convert TensorFlow tensors -> NumPy arrays -> PyTorch tensors
    enc_in_pt = torch.tensor(encoder_input.numpy(), dtype=torch.long)
    dec_in_pt = torch.tensor(decoder_input.numpy(), dtype=torch.long)
    dec_tar_pt = torch.tensor(decoder_target.numpy(), dtype=torch.long)

    return enc_in_pt, dec_in_pt, dec_tar_pt, src_vectorizer, trg_vectorizer, max_src_len, max_trg_len


def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    model.train() 
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for batch_idx, (src_data, trg_in_data, trg_out_data) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass uses 'trg_in_data' (which has startseq)
            output, _ = model(encoder_input=src_data, decoder_input=trg_in_data, training=True)
            
            # Reshape for CrossEntropyLoss
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim) 
            
            # Target uses 'trg_out_data' (which has endseq)
            target = trg_out_data.reshape(-1) 
            
            loss = criterion(output, target)
            
            # 5. Backward pass
            loss.backward()
            
            # 6. Gradient Clipping (Highly recommended for LSTMs to prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 7. Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Print average loss for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss / len(train_loader):.4f}")


def translate_sentence(model, sentence, src_vectorizer, trg_vectorizer, max_len=50):
    # 1. Put the model in evaluation mode (turns off dropout, batchnorm, etc.)
    model.eval()

    # 2. Preprocess the raw string using your Keras vectorizer
    # We wrap it in a list because the vectorizer expects a batch
    input_vector = src_vectorizer(np.array([sentence]))
    
    # 3. Convert to PyTorch tensor and move to GPU/CPU
    input_tensor = torch.tensor(input_vector.numpy(), dtype=torch.long)

    # 4. Turn off gradient tracking (saves memory and speeds up inference)
    with torch.no_grad():
        # Pass training=False to trigger your custom inference loop!
        output_tokens, _ = model(encoder_input=input_tensor, training=False, max_trg_len=max_len)

    # output_tokens shape is [1, max_len]. We squeeze it to a 1D list of numbers.
    output_tokens = output_tokens.squeeze().cpu().numpy()

    # 5. Convert token IDs back to words
    vocab = trg_vectorizer.get_vocabulary()
    translated_words = []
    
    for token_id in output_tokens:
        word = vocab[token_id]
        
        # Stop translating if the model outputs the end token
        if word == 'endseq':
            break
            
        # Ignore padding or unknown tokens in the final output
        if word not in ['', '<PAD>', '[UNK]', 'startseq']:
            translated_words.append(word)

    # Join the list of words into a single string
    return " ".join(translated_words)

__all__ = ["load_and_vectorize_data", "train_model", "translate_sentence"]