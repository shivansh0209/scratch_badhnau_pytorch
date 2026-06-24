import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, src_vectorizer, trg_vectorizer):
        super().__init__()

        self.state_size = 128
        self.embedding_dim = 64
        
        # Save this to class so we can access word_to_index in forward()
        self.trg_vectorizer = trg_vectorizer 

        self.encoder_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.state_size, batch_first=True, bias=True)
        
        # Decoder input size is embedding_dim + context_vector size (state_size)
        self.decoder = nn.LSTM(input_size=self.embedding_dim + self.state_size, hidden_size=self.state_size, batch_first=True, bias=True)

        self.encoder_embedding = nn.Embedding(num_embeddings=src_vectorizer.vocab_size, embedding_dim=self.embedding_dim)
        self.decoder_embedding = nn.Embedding(num_embeddings=trg_vectorizer.vocab_size, embedding_dim=self.embedding_dim)

        # Attention output must be 1 to calculate scores
        self.attention = nn.Linear(2 * self.state_size, 1)
        
        # Added the final fully connected projection layer
        self.fc_out = nn.Linear(self.state_size, trg_vectorizer.vocab_size)

    # Added max_trg_len as an argument so the inference loop knows when to stop
    def forward(self, encoder_input, decoder_input=None, training=False, max_trg_len=50):
        batch_size = encoder_input.size(0)
        
        encoder_embedded = self.encoder_embedding(encoder_input)
        encoder_output, (hidden, cell) = self.encoder_lstm(encoder_embedded)
        prev_decoder_state = (hidden, cell)
        
        # We need the encoder sequence length to properly shape our attention tensors
        seq_len = encoder_output.size(1)

        if training:
            # List to store predictions for the whole sentence
            all_predictions = [] 
            
            for word_idx in range(decoder_input.size(1)):
                decoder_word_input = decoder_input[:, word_idx].unsqueeze(1)
                
                # The Shape Fix. Expand the decoder hidden state from [1, batch, 128] 
                # to [batch, seq_len, 128] so it matches encoder_output for concatenation.
                hidden_expanded = prev_decoder_state[0].transpose(0, 1).repeat(1, seq_len, 1)
                
                attention_input = torch.cat((hidden_expanded, encoder_output), dim=2)
                attention_weights = torch.softmax(self.attention(attention_input), dim=1)
                context_vector = torch.sum(attention_weights * encoder_output, dim=1, keepdim=True)

                decoder_embedded = self.decoder_embedding(decoder_word_input)
                decoder_embedded = torch.cat((context_vector, decoder_embedded), dim=2)
                decoder_output, (hidden, cell) = self.decoder(decoder_embedded, prev_decoder_state)
                prev_decoder_state = (hidden, cell)

                # Pass through final projection layer and save it
                prediction = self.fc_out(decoder_output)
                all_predictions.append(prediction)
                
            # Concatenate list of [batch, 1, vocab_size] into [batch, seq_len, vocab_size]
            return torch.cat(all_predictions, dim=1), prev_decoder_state

        else:
            start_token = self.trg_vectorizer.word_to_index['startseq']
            # Made device dynamic so it works on GPU if your inputs are on GPU
            decoder_word_input = torch.full((batch_size, 1), start_token, dtype=torch.long)
            decoded_words = []

            for word_idx in range(max_trg_len):
                # The Shape Fix for inference
                hidden_expanded = prev_decoder_state[0].transpose(0, 1).repeat(1, seq_len, 1)

                attention_input = torch.cat((hidden_expanded, encoder_output), dim=2)
                attention_weights = torch.softmax(self.attention(attention_input), dim=1)
                context_vector = torch.sum(attention_weights * encoder_output, dim=1, keepdim=True)

                decoder_embedded = self.decoder_embedding(decoder_word_input)
                decoder_embedded = torch.cat((context_vector, decoder_embedded), dim=2)
                decoder_output, (hidden, cell) = self.decoder(decoder_embedded, prev_decoder_state)
                prev_decoder_state = (hidden, cell)

                # Pass through final layer BEFORE doing argmax
                prediction = self.fc_out(decoder_output)
                predicted_word_idx = torch.argmax(prediction[:, -1, :], dim=1).unsqueeze(1)
                
                decoder_word_input = predicted_word_idx
                decoded_words.append(predicted_word_idx)
                
            # Un-indented the return statement so the loop actually finishes!
            # Returns a tensor of shape [batch, max_trg_len] containing predicted token IDs
            return torch.cat(decoded_words, dim=1), prev_decoder_state


__all__ = ['LSTMModel']
