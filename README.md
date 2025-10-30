# mini-transformer-lab

A compact transformer-based language model built from scratch in pure PyTorch — designed for learning, tinkering, and really understanding how LLMs work under the hood.

> 🔬 **Built with curiosity, debugged with persistence**
> 
> Almost every error and bug in this project was debugged (sometimes painfully) with a bit of help from AI tools — and yeah, my brain had its fair share of contributions too in debugging, coding, and the whole idea. Because honestly, what else would you expect from a self-taught Python programmer building their own LLM?

## 🎯 What's Inside

- 🧠 **mini-transformer-lab.py** — Main model & training script  
- 📚 **example_data/** — Sample texts to train on  
- 🎛️ **checkpoints/** — Saved model weights  
- 🔬 **experiments/** — Training logs & results


## 🚀 Quick Start

### Installation

```bash
# Clone this repository
git clone https://github.com/Aranya-Marjara/mini-transformer-lab.git
cd mini-transformer-lab

# Install dependencies (just PyTorch!)
pip install torch (installing in a virtual environtment is highly recommended)
```
## Create a file
nano my_data.txt

## Paste something You want to train here is an example:
```
Artificial intelligence has changed the world of technology forever. From simple rule-based systems to advanced large language models, AI continues to evolve with astonishing speed. The transformer architecture revolutionized the way machines understand language, allowing them to capture long-range dependencies and contextual meaning with ease.

Understanding how these models work under the hood is a fascinating challenge. Each token, attention head, and layer contributes to the model's ability to reason, summarize, and generate text. Building a small transformer model from scratch is not just an engineering task but also an educational journey into the core mechanics of intelligence.

Training even a mini transformer on your own data helps reveal how neural networks learn from patterns. As the loss goes down, the model starts to grasp structure — words begin to align, and meaning starts to emerge. These tiny experiments reflect, in miniature, what the biggest LLMs in the world are doing at scale.

Open-source research and tinkering allow anyone to learn how these systems work. Sharing your code publicly means that others can build on your work, improve it, and contribute ideas. True progress happens when knowledge is free, transparent, and collaborative.

This text exists to provide enough data for your mini-transformer-lab model to train successfully, test attention layers, and generate a few coherent lines. Keep experimenting, because every small step adds up to something bigger in the world of AI.
```


## Basic Usage
# 🏋️ Train a new model on your text
python3 mini-transformer-lab.py train --data your_text.txt --epochs 10

# 🎲 Generate some text
python3 mini-transformer-lab.py generate --checkpoint checkpoint.pt --prompt "The future of AI is"

## Transformer Architecture

```Input: "Hello world"
     ↓
Tokenize: [23, 45, 12, 67]
     ↓
Embeddings + Positional Encoding
     ↓
┌─────────────────────────────────┐
│  Multi-Head Self-Attention      │ ←──┐
│  (Causal Masking)               │    │ Repeat 4x
│  LayerNorm + Feed Forward       │ ←──┘
└─────────────────────────────────┘
     ↓
Output Logits → Next Token Prediction
```

## Sampling Strategies

```Temperature Sampling:
   logits = logits / temperature
   ┌─ Hot (0.8) → Creative
   ├─ Warm (1.0) → Balanced  
   └─ Cold (1.5) → Conservative

Top-k Sampling:
   ┌─ Keep only top k tokens
   └─ k=50 → Diverse but coherent

Top-p (Nucleus) Sampling:
   ┌─ Keep tokens until cumulative
   │  probability reaches p
   └─ p=0.9 → Dynamic vocabulary
```
## Training Flow
```
📖 Load Text Data
   ↓
🔡 Tokenize Characters
   ↓
🔄 For each epoch:
   ├── 🎲 Sample training batch
   ├── 🧠 Forward pass
   ├── 📉 Compute loss
   ├── ↪️ Backward pass
   └── ⚙️ Update weights
   ↓
💾 Save checkpoint
```
## Advanced Usage (Training with Custom Parameters)
```
python mini-transformer-lab.py train \
  --data shakespeare.txt \
  --epochs 20 \
  --batch_size 32 \
  --context_length 256 \
  --lr 0.0003 \
  --device cuda  # Use GPU if available
```
## Creative Generation
```
# Temperature controls creativity
python mini-transformer-lab.py generate \
  --checkpoint best_model.pt \
  --prompt "In a galaxy far away" \
  --temperature 0.7 \
  --top_k 40 \
  --top_p 0.9 \
  --max_new_tokens 200
```
## Resume Training
```
python mini-transformer-lab.py train \
  --data my_novel.txt \
  --resume checkpoint_epoch_15.pt \
  --epochs 25
```
## Model Architecture
```
MiniTransformer Config:
├── 📏 Context Length: 256 tokens
├── 🎯 Model Dimension: 256
├── 👥 Attention Heads: 8
├── 🏗️ Layers: 4
├── 🧠 Feed Forward: 1024
└── 💧 Dropout: 0.1
```
## Design Philosophy
```
🧠 Understanding > Scale
   └── Built to learn, not to compete with GPT Models

🔧 Simplicity > Complexity  
   └── Pure PyTorch, no fancy trainers

🎯 Education > Production
   └── Readable code with clear explanations

🐛 Debugging > First-try perfection
   └── Every error was a learning opportunity
```
