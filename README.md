# 🧠 mini-transformer-lab  
*A compact transformer-based language model built from scratch in pure PyTorch.*

> "LLMs aren’t magic. They’re just math — and I wanted to build one myself."

---

![Transformer Header](https://github.com/Aranya-Marjara/mini-transformer-lab/assets/your-gif-id-here/transformer-banner.gif)

<p align="center">
  <a href="https://github.com/Aranya-Marjara/mini-transformer-lab/stargazers">
    <img src="https://img.shields.io/github/stars/Aranya-Marjara/mini-transformer-lab?color=gold" />
  </a>
  <a href="https://github.com/Aranya-Marjara/mini-transformer-lab/forks">
    <img src="https://img.shields.io/github/forks/Aranya-Marjara/mini-transformer-lab?color=lightblue" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/License-GPLv3-green.svg" />
</p>

---

## 🚀 What is This?

`mini-transformer-lab` is a **from-scratch Transformer model**, written in **pure PyTorch**, built to learn how real LLMs like GPT work under the hood — token by token, layer by layer.

Every tensor operation is transparent.  
Every gradient is visible.  
And every bug taught me more than any tutorial could.

---

## 🧩 Features

- 🧠 Self-Attention (multi-head, causal masking)
- 🔡 Character-level tokenizer
- 💾 Checkpoint saving & resuming
- 🎛️ Configurable hyperparameters (context length, layers, heads, etc.)
- 🪄 Top-k / Top-p sampling for creative text
- 💬 Command-line interface (train + generate)

---

## ⚙️ Installation

```bash
git clone https://github.com/Aranya-Marjara/mini-transformer-lab.git
cd mini-transformer-lab
pip install torch tqdm
