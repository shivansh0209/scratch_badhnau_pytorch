# Seq2Seq LSTM Translation Model with Bahdanau Attention

An implementation of a Sequence-to-Sequence (Seq2Seq) English-to-French language translator built from scratch in PyTorch, utilizing a Keras text preprocessing pipeline and a custom time-step loop for explicit dynamic attention orchestration.

## Features

* **Encoder-Decoder Architecture:** Built using multi-unit LSTM layers to mitigate the vanishing gradient problem inherent in traditional RNNs.
* **Custom Bahdanau (Additive) Attention Layer:** Implemented step-by-step tensor math to calculate alignment scores dynamically at every decoding time-step.
* **Hybrid Framework Pipeline:** Consumes fast, optimized Keras `TextVectorization` pipelines directly into a PyTorch training loop via custom memory adapters.
* **Autoregressive Inference & Teacher Forcing:** Features a distinct operational mode switch for standard cross-entropy optimization vs. greedy argmax generation.


---

## Tensor Shape Lifecycle (Batch Size = B)

| Variable | Tensor Shape Mapping | Architectural Stage |
| :--- | :--- | :--- |
| `encoder_input` | $(B, \text{Seq\_Len}_{\text{src}})$ | Raw Token IDs fed into Encoder |
| `encoder_output` | $(B, \text{Seq\_Len}_{\text{src}}, 128)$ | All Hidden Sequence States for Attention |
| `hidden_expanded` | $(B, \text{Seq\_Len}_{\text{src}}, 128)$ | Decoder state broadcasted across input space |
| `attention_input` | $(B, \text{Seq\_Len}_{\text{src}}, 256)$ | Combined contextual alignment matrix |
| `context_vector` | $(B, 1, 128)$ | Weighted context sum matching state footprint |
| `decoder_embedded` | $(B, 1, 192)$ | Context concatenated with targeted token embedding |
| `prediction` | $(B, 1, \text{Vocab}_{\text{trg}})$ | Raw vocab scores generated per time-step |

---

## Getting Started

### Prerequisites
Make sure you have both TensorFlow and PyTorch installed inside your active environment:
```bash
pip install -r reqs.txt
python 01_lstm_model.ipynb