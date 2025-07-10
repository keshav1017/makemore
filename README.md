# 🔤 Char-Level Language Modeling from Scratch

> A step-by-step, hands-on deep learning journey into the inner workings of language models — built entirely from scratch with PyTorch and Python.

---

## 🧠 Overview

This project is a complete replication and learning implementation of Andrej Karpathy’s excellent [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) lecture series. The goal is to **build a character-level language model** starting from bigrams all the way to deep neural architectures — using nothing but PyTorch, basic math, and curiosity.

Each notebook is a **self-contained part**, and together they evolve into a more powerful model step-by-step — while teaching you core concepts like tensor operations, neural networks, gradient flows, batch normalization, and more.

---

## 📚 Project Structure

| Notebook               | Description                                                                |
| ---------------------- | -------------------------------------------------------------------------- |
| `01_first_part.ipynb`  | Bigram language model using torch.Tensors and negative log-likelihood loss |
| `02_second_part.ipynb` | MLP language model with training loop, hyperparameter tuning, eval logic   |
| `03_third_part.ipynb`  | Deep dive into forward/backward passes, activation stats, BatchNorm        |
| `04_fourth_part.ipynb` | Manual backprop through the entire model without `loss.backward()`         |
| `05_part_five.ipynb`   | Deeper MLP with WaveNet-style tree structure using `torch.nn`              |

---

## 🔢 Part-by-Part Breakdown

### ✅ **Part 1: Bigram Language Model**

* Introduced **torch.Tensors** and their nuances
* Built a basic bigram model (predict next char from current)
* Explained **language modeling loop**: training, sampling, evaluating loss
* Introduced **negative log likelihood** as the loss function

**Key Learnings:**

* Tensor indexing and broadcasting
* Efficient training without any PyTorch autograd features
* Simple sampling-based text generation

---

### ✅ **Part 2: MLP Character-Level Language Model**

* Replaced the bigram logic with a **Multilayer Perceptron (MLP)**
* Introduced:

  * Model training with autograd
  * Manual training loop
  * Learning rate tuning
  * Overfitting/underfitting intuition

**Key Learnings:**

* How to train simple MLPs
* Dataset splitting (train/dev/test)
* Importance of initialization and batch processing

---

### ✅ **Part 3: Deep Dive into MLP Internals**

* Analyzed statistics of:

  * Forward pass activations
  * Backward gradients
* Identified training pitfalls (e.g., exploding/vanishing gradients)
* Introduced **Batch Normalization** to stabilize training

**Key Learnings:**

* Diagnostic tools for network health
* Scaling and activation distribution
* Role of normalization in deep networks

---

### ✅ **Part 4: Manual Backpropagation**

* **Backpropagated manually** through:

  * Cross-entropy loss
  * Linear layers
  * BatchNorm
  * Embedding layers
* Avoided using `loss.backward()` entirely

**Key Learnings:**

* Intuitive understanding of gradient flow
* How each layer contributes to the backward pass
* Efficiency of tensor-level operations vs scalar autograd (like in micrograd)

---

### ✅ **Part 5: Deepening the MLP (WaveNet-style Tree)**

* Made the model **deeper** with **hierarchical structure**
* Resulting architecture resembles **WaveNet (2016)** from DeepMind
* Introduced:

  * Tree-style design
  * Causal convolution idea
  * `torch.nn.Module` best practices

**Key Learnings:**

* Building deeper networks systematically
* Understanding tensor shape transformations
* Working efficiently with PyTorch modules

---

## 🎯 Skills & Concepts Practiced

* 🧮 PyTorch tensors, autograd, and low-level API usage
* 🧠 Model architecture design from scratch
* 🔁 Backpropagation, both automatic and manual
* 🧪 Diagnostics: activation/gradient statistics
* 🛠 Batch normalization, embeddings, and MLPs
* 🧱 Building scalable deep learning models

---

## 🖥️ How to Run

> Make sure you have Python 3.8+ and PyTorch installed.

```bash
pip install torch matplotlib numpy
jupyter notebook 01_first_part.ipynb
```

You can step through each notebook sequentially from part 1 to part 5.

---

## 🧩 Sample Output (Coming Soon)

> Add generated sample text, model loss curve plots, or architecture diagrams here.

---

## 🙌 Credits

* Original idea and content by [Andrej Karpathy](https://github.com/karpathy)
* This project is a **learning-based replication** for educational purposes

---

## 📌 Final Thoughts

This project has been an excellent exercise in deeply understanding:

* The building blocks of deep learning
* Why modern architectures work
* How you can write a neural network from scratch and *trust it*

If you’re learning deep learning, **there’s no better way to understand it than to build it from the ground up.**
