# RNN Language Model from Scratch

A 3-layer stacked Recurrent Neural Network (RNN) language model built entirely from scratch using **CuPy** (GPU-accelerated NumPy), trained on the **Penn Treebank** corpus. No PyTorch, no TensorFlow — just raw matrix math and backpropagation through time (BPTT).

---

## What This Project Does

This model learns to predict the next word in a sentence — the same core objective that powers modern language models. Given a sequence of words, it outputs a probability distribution over the entire vocabulary for the next token.

**Training corpus:** Penn Treebank (via NLTK) — 3,914 Wall Street Journal sentences  
**Vocabulary size:** 12,410 unique tokens  
**Architecture:** Embedding → RNN Layer 1 → RNN Layer 2 → RNN Layer 3 → Softmax Output  
**Backend:** CuPy (CUDA GPU)

---

## The Key Insight This Project Resolved

### Hidden Layers vs. Hidden States — A Common Point of Confusion

When you first learn about neural networks, you encounter **hidden layers**: intermediate layers in a feedforward network that transform inputs into richer representations. They are static — given the same input, they always produce the same output.

RNNs introduce a fundamentally different concept: the **hidden state**. This project made that distinction concrete.

In a regular feedforward network:
```
input → [hidden layer] → output
```
The hidden layer is just a function. It has no memory.

In an RNN, the hidden state is updated at every timestep and carries information *forward through time*:
```
input_t + h_{t-1} → h_t → output_t
```

The hidden state `h_t` is not just a layer — it is the model's **memory of everything it has seen so far**. Each new word is processed in the context of all previous words, encoded in that single vector.

This project makes this tangible: `h1_prev`, `h2_prev`, `h3_prev` are explicitly passed forward at every timestep in the forward pass, and the BPTT backward pass propagates gradients all the way back through those states. There is no "hidden layer" in the feedforward sense — there is a hidden *state* that evolves through the sequence.

The stacking of three RNN layers adds depth at each timestep, so you get both:
- **Depth** (3 layers of transformation at each word)
- **Memory** (hidden states carrying context across words)

These are orthogonal concepts, and implementing both by hand makes the distinction impossible to confuse again.

---

## Architecture

```
Word indices (x)
      │
      ▼
 Embedding E          [vocab_size × embedding_size]
      │
      ▼
 RNN Layer 1          h1_t = tanh(W_xh1 · x_t  +  W_hh1 · h1_{t-1}  +  b_h1)
      │
      ▼
 RNN Layer 2          h2_t = tanh(W_xh2 · h1_t  +  W_hh2 · h2_{t-1}  +  b_h2)
      │
      ▼
 RNN Layer 3          h3_t = tanh(W_xh3 · h2_t  +  W_hh3 · h3_{t-1}  +  b_h3)
      │
      ▼
 Output Layer         y_t  =  softmax(W_hy · h3_t  +  b_y)
      │
      ▼
 Next-word probability distribution
```

### Parameters

| Component | Shape | Description |
|-----------|-------|-------------|
| `E` | `[12410 × 128]` | Word embeddings |
| `W_xh1`, `W_hh1`, `b_h1` | `[128×128]`, `[128×128]`, `[128×1]` | RNN Layer 1 |
| `W_xh2`, `W_hh2`, `b_h2` | `[128×128]`, `[128×128]`, `[128×1]` | RNN Layer 2 |
| `W_xh3`, `W_hh3`, `b_h3` | `[128×128]`, `[128×128]`, `[128×1]` | RNN Layer 3 |
| `W_hy`, `b_y` | `[12410×128]`, `[12410×1]` | Output projection |

---

## How It Relates to Modern Decoder-Only LLMs

This project is architecturally closer to GPT than it might seem. The connection is not superficial.

| Property | This RNN LM | GPT / Decoder-only Transformer |
|----------|-------------|-------------------------------|
| **Objective** | Predict next token | Predict next token |
| **Autoregressive** | Yes — processes left to right | Yes — causal attention mask |
| **Context** | Hidden state (compressed) | Full attention over past tokens |
| **Depth** | 3 stacked layers | Many stacked blocks |
| **Embeddings** | Learned `E` matrix | Learned token + positional embeddings |
| **Output** | Softmax over full vocabulary | Softmax over full vocabulary |
| **Training signal** | Cross-entropy on next word | Cross-entropy on next word |
| **Tokenization** | Word-level | Subword (BPE/WordPiece) |

The critical difference is *how* past context is encoded. This RNN compresses everything seen so far into a fixed-size hidden state vector — a lossy bottleneck. A Transformer's self-attention mechanism can directly attend to any past token, removing that bottleneck entirely.

But the **training objective, the autoregressive generation loop, the embedding table, and the output projection to vocabulary logits** are identical in spirit. This model is a direct ancestor of the decoder-only LLM architecture.

---

## Training Details

- **Data split:** 80% train / 10% validation / 10% test
- **Special tokens:** `<BOS>` and `<EOS>` added to each sentence
- **Optimizer:** Vanilla SGD (`lr = 0.001`)
- **Gradient clipping:** Global norm clipping at `max_norm = 5.0` (critical for deep RNNs trained with plain SGD)
- **Epochs:** 50
- **Loss function:** Cross-entropy (per-token average)

### Training Results

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 8.6450 | 7.7448 |
| 10 | 7.0238 | 7.0737 |
| 25 | 6.9230 | 7.0088 |
| 50 | 6.8823 | 6.9948 |

Loss converges steadily. The train/val gap is narrow, indicating the model is learning structure from the corpus rather than memorizing.

---

## Setup and Usage

### Requirements

```bash
pip install cupy-cuda12x nltk
```

> Requires a CUDA-compatible GPU. Adjust `cupy-cuda12x` to match your CUDA version.

### Data

```python
import nltk
nltk.download('treebank')
```

### Run

Open the notebook in Google Colab (T4 GPU recommended) and run cells top to bottom. The full training loop runs in the final cell.

---

## Project Structure

```
notebook.ipynb        # Everything — data prep, model, training, evaluation
README.md             # This file
```

---

## What's Implemented by Hand

- Word-to-index vocabulary construction
- BOS/EOS token insertion
- Train/val/test splitting
- Forward pass through embedding + 3-layer stacked RNN + softmax
- Full BPTT (backpropagation through time) across all three layers
- Gradient accumulation across timesteps
- Global gradient norm clipping
- Vanilla SGD parameter update
- Per-epoch validation evaluation

Nothing from a deep learning framework was used. Every gradient is derived and applied manually.
