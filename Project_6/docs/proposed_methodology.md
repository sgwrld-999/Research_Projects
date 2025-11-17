# Proposed Approach and Methodology

This document outlines the comprehensive methodology for Project-6, focusing on the development of a lightweight, Transformer-based Intrusion Detection System (IDS) for Industrial Internet of Things (IIoT) environments. The methodology is structured into three primary sections: a foundational review of the Transformer architecture, a detailed analysis of the datasets employed, and a formal description of the proposed LITE framework.

---

## 1. Transformer Architecture

This section provides a foundational understanding of the Transformer model, beginning with the seminal "Attention Is All You Need" paper. We dissect its core components and analyze its computational complexity, establishing the basis for our proposed lightweight architecture.

### 1.1. Background: The "Attention Is All You Need" Paradigm

The Transformer, introduced by Vaswani et al. (2017), marked a paradigm shift in sequence transduction tasks, moving away from the recurrent and convolutional structures that were previously dominant. By eliminating recurrence and relying solely on a self-attention mechanism, the Transformer achieved state-of-the-art performance in machine translation while requiring significantly less training time and being more parallelizable.

The core innovation is the **attention mechanism**, which allows the model to weigh the importance of different words (or, in our case, features) in the input sequence when processing a given element. This capability to draw global dependencies between input and output is what gives the Transformer its power.

#### 1.1.1. The Attention Mechanism: Q, K, V

The attention mechanism can be described as mapping a query and a set of key-value pairs to an output. The query (Q), keys (K), and values (V) are vector representations derived from the input embeddings. The output is computed as a weighted sum of the values, where the weight assigned to each value is determined by the compatibility of the query with the corresponding key.

The most common form is **Scaled Dot-Product Attention**, defined by the equation:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The components are:
- **Q (Query)**: A matrix of `n_q` query vectors, each of dimension `d_k`. This represents the current token/feature being processed.
- **K (Keys)**: A matrix of `n_k` key vectors, each of dimension `d_k`. These are "probes" associated with each token in the sequence that the query can interact with.
- **V (Values)**: A matrix of `n_k` value vectors, each of dimension `d_v`. These contain the actual information of the tokens.
- **`QK^T`**: The dot product computes a similarity score between each query and every key. A higher score implies greater relevance.
- **`√d_k` Scaling Factor**: The dot products can grow large in magnitude, pushing the softmax function into regions with extremely small gradients. Scaling by the square root of the key dimension (`d_k`) counteracts this effect, leading to more stable training.
- **Softmax**: This function is applied to the scaled scores to obtain a set of positive attention weights that sum to one. These weights represent the distribution of focus for a given query.
- **`...V`**: The final output is the weighted sum of the value vectors, where the weights are the computed attention scores. This allows the model to focus on the most relevant parts of the input sequence.

#### 1.1.2. Self-Attention, Multi-Head Attention, and Cross-Attention

- **Self-Attention**: This is a special case of attention where the Q, K, and V vectors are all derived from the same input sequence. It allows each position in the sequence to attend to all other positions, capturing intra-sequence dependencies. This is the cornerstone of the Transformer's Encoder.

- **Multi-Head Attention (MHA)**: Instead of performing a single attention function, MHA runs the Scaled Dot-Product Attention mechanism multiple times in parallel. The input Q, K, and V are linearly projected into `h` different subspaces (where `h` is the number of "heads"). Attention is computed for each head, and the resulting output vectors are concatenated and linearly projected back to the original dimension.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$
$$
\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

MHA allows the model to jointly attend to information from different representational subspaces at different positions. A single attention head might learn to focus on syntactic relationships, while another focuses on semantic ones.

- **Cross-Attention**: This mechanism is used in the Decoder part of the Transformer. Here, the queries (Q) come from the previous decoder layer, while the keys (K) and values (V) come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence, which is crucial for sequence-to-sequence tasks like translation, where the model needs to align output words with relevant input words.

#### 1.1.3. Encoder and Decoder Architecture

The original Transformer is an Encoder-Decoder stack.

- **Encoder**: The encoder's role is to map an input sequence of symbol representations `(x_1, ..., x_n)` to a sequence of continuous representations `z = (z_1, ..., z_n)`.
  - It consists of a stack of `N` identical layers.
  - Each layer has two sub-layers:
    1. A **Multi-Head Self-Attention** mechanism.
    2. A simple, position-wise **Feed-Forward Network (FFN)**.
  - A **residual connection** followed by **Layer Normalization** is applied around each of the two sub-layers. The output of each sub-layer is `LayerNorm(x + Sublayer(x))`.

- **Decoder**: The decoder is also composed of a stack of `N` identical layers. Its role is to generate an output sequence `(y_1, ..., y_m)` one symbol at a time, using the encoder's output `z`.
  - In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer:
    1. A **Masked Multi-Head Self-Attention** mechanism. The masking ensures that predictions for position `i` can depend only on the known outputs at positions less than `i`.
    2. A **Multi-Head Cross-Attention** mechanism that attends to the output of the encoder stack.
    3. A position-wise **Feed-Forward Network**.
  - Residual connections and Layer Normalization are also applied to each sub-layer.

#### 1.1.4. Architectural Variants and Use Cases

- **Encoder-Decoder**: The full architecture (e.g., T5, BART) is ideal for sequence-to-sequence tasks where the input and output can have different lengths and structures, such as machine translation or text summarization.
- **Encoder-Only**: These models (e.g., BERT, RoBERTa) build rich, bidirectional representations of the input. They are pre-trained on tasks like masked language modeling and are highly effective for downstream tasks that require a deep understanding of the input context, such as text classification, sentiment analysis, or named entity recognition. Our work leverages this architecture for intrusion detection, treating it as a classification problem.
- **Decoder-Only**: These models (e.g., GPT series) are auto-regressive and are trained to predict the next token in a sequence. They are powerful generative models used for tasks like text generation, dialogue systems, and code completion.

### 1.2. Time and Space Complexity

The computational complexity of the Transformer is a critical consideration, especially for resource-constrained environments like the IIoT edge. Let `n` be the sequence length and `d` be the model's hidden dimension.

- **Time Complexity**: The dominant computation is the matrix multiplication `QK^T` within the self-attention mechanism. This involves multiplying a matrix of size `(n x d_k)` by one of size `(d_k x n)`, resulting in a complexity of **`O(n^2 * d)`**. The feed-forward layer has a complexity of `O(n * d^2)`, but since `d` is typically fixed and smaller than `n` in many applications, the `n^2` term from the attention layer is the primary bottleneck.

- **Space Complexity**: The space complexity is also dictated by the attention mechanism, which needs to store the attention matrix of size `(n x n)`. This results in a memory requirement of **`O(n^2)`**.

This quadratic dependency on the sequence length makes the standard Transformer computationally expensive for very long sequences, motivating the research into more efficient, "lightweight" variants like the one proposed in this project.

---

## 2. Dataset Analysis and Preprocessing

The robustness and generalizability of an IDS are heavily dependent on the quality and diversity of the training data. We utilize two contemporary, large-scale IIoT datasets to ensure our model is evaluated against a wide range of realistic network traffic and attack scenarios.

### 2.1. Dataset Introduction

#### 2.1.1. Edge-IIoTset

The Edge-IIoTset is a comprehensive, realistic dataset designed to represent modern IoT and IIoT network traffic. It includes data from a wide variety of devices, such as soil moisture sensors, heart rate sensors, and smart thermostats, as well as malicious traffic generated from various attack vectors (e.g., DDoS, SQL injection, port scanning). Its key advantage is the inclusion of 14 distinct attack types, providing a fine-grained benchmark for multi-class classification.

#### 2.1.2. CIC-IoT-2023 (CICIoT)

The CIC-IoT-2023 dataset is another large-scale, modern dataset created to address the shortcomings of older IDS datasets. It was generated from a realistic IoT testbed containing a mix of IoT devices and includes both benign traffic and a diverse set of 33 attacks. The attacks range from traditional network attacks (DDoS, DoS) to more sophisticated, IoT-specific exploits. The dataset's scale and diversity make it an excellent resource for training and validating high-performance IDS models.

### 2.2. Preprocessing Pipeline

Raw network data is unsuitable for direct use in machine learning models. A multi-stage preprocessing pipeline was designed to clean, transform, and normalize the data.

#### 2.2.1. Edge-IIoT Preprocessing

1.  **Data Cleaning**: Redundant or irrelevant columns (e.g., `frame.time`, `ip.src_host`, `ip.dst_host`) were removed. Missing values and infinite values (`NaN`, `inf`) generated during feature extraction were imputed using the mean of their respective columns.
2.  **Categorical Encoding**: The `icmp.type` column, being categorical, was one-hot encoded to convert it into a numerical format.
3.  **Label Encoding**: The multi-class attack labels were mapped to integer values. For binary classification, all 14 attack types were mapped to `1` (Attack) and the `Normal` class was mapped to `0`.
4.  **Normalization**: All numerical features were scaled to the range `[0, 1]` using Min-Max scaling. This is crucial for deep learning models to ensure stable training and prevent features with large magnitudes from dominating the learning process.

#### 2.2.2. CICIoT Preprocessing

The CICIoT dataset underwent a similar preprocessing pipeline:
1.  **Data Cleaning**: Columns with a single unique value or a high percentage of missing values were dropped.
2.  **Label Encoding**: The 33 attack types were mapped to integer values for multi-class classification. For the binary task, they were consolidated into a single "Attack" class.
3.  **Normalization**: Min-Max scaling was applied to all numerical features to bring them into the `[0, 1]` range.

### 2.3. Feature Selection

Not all features contribute equally to the detection of intrusions. Feature selection helps reduce model complexity, decrease training time, and potentially improve performance by eliminating noise.

#### 2.3.1. Edge-IIoT Feature Selection

For the Edge-IIoT dataset, we employed a **Correlation-based Feature Selection (CFS)** method. Features that have a high correlation with the target label but low correlation with each other are selected. This approach effectively reduces redundancy while retaining predictive power. Based on this analysis, a subset of the most informative features was selected for model training.

#### 2.3.2. CICIoT Feature Selection

A similar CFS approach was applied to the CICIoT dataset. Given its larger feature space, feature selection was critical for managing the computational load. The process identified a reduced set of features that provided the best trade-off between model complexity and predictive accuracy.

---

## 3. Proposed Methodology: LITE Framework

To address the computational demands of the standard Transformer, we propose **LITE (Lightweight Intrusion Transformer for Edge)**, a novel architecture optimized for efficiency and performance in IIoT environments.

### 3.1. LITE: Proposed Architecture Overview

LITE is an **Encoder-only** Transformer architecture, as intrusion detection is fundamentally a classification task that requires a deep understanding of the input features. The core of LITE is a modified self-attention mechanism designed to reduce the quadratic complexity of the standard model.

Our proposed attention mechanism is a form of **sparse attention**. Instead of allowing each feature to attend to every other feature (global attention), we restrict the attention to a smaller, localized window. This is based on the hypothesis that for network intrusion detection, the most critical interactions are between a feature and its immediate neighbors in the feature vector, rather than distant ones.

The architecture consists of:
1.  **Embedding Layer**: The input features (a vector of size `d_features`) are passed through a linear layer to project them into the model's hidden dimension, `d_model`. Positional encodings are added to give the model information about the relative position of features.
2.  **LITE Encoder Stack**: A stack of `N` LITE Encoder layers. Each layer contains:
    - A **Sparse Multi-Head Self-Attention** mechanism.
    - A position-wise Feed-Forward Network.
    - Residual connections and Layer Normalization.
3.  **Classification Head**: The output from the final LITE Encoder layer corresponding to the `[CLS]` token (a special token prepended to the sequence) is passed through a linear layer followed by a softmax function to produce the final class probabilities (e.g., Normal vs. Attack).

### 3.2. Computational Complexity Analysis

The key innovation of LITE is its reduced complexity. By using a sparse attention mechanism with a fixed window size `w`, the complexity of the attention computation is reduced significantly.

- **Time Complexity**: The time complexity of the sparse attention mechanism is **`O(n * w * d)`**, where `w` is the window size and `n` is the sequence length. Since `w << n`, this is a substantial improvement over the `O(n^2 * d)` of the standard Transformer. This makes LITE much faster to train and suitable for real-time inference on edge devices.

- **Space Complexity**: Similarly, the space required to store the attention scores is reduced to **`O(n * w)`**, a linear relationship with the sequence length, compared to the quadratic `O(n^2)` of standard attention.

### 3.3. Experimentation Setup

All experiments were conducted on a local system with the following configuration to simulate a moderately powerful edge computing environment.

- **Hardware**:
  - **CPU**: Intel Core i9-12900K
  - **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
  - **RAM**: 64GB DDR5

- **Software & Tool Stack**:
  - **OS**: Windows 11
  - **Programming Language**: Python 3.9
  - **Core Libraries**:
    - PyTorch (for model development and training)
    - Scikit-learn (for preprocessing and evaluation metrics)
    - Pandas (for data manipulation)
    - NumPy (for numerical operations)
    - Hydra (for configuration management)

- **Hyperparameters**:
  - **Model Dimension (`d_model`)**: 128
  - **Number of Encoder Layers (`N`)**: 4
  - **Number of Attention Heads (`h`)**: 4
  - **Learning Rate**: 1e-4 (with Adam optimizer)
  - **Batch Size**: 256
  - **Epochs**: 50 (with early stopping)

This setup ensures reproducibility and provides a clear baseline for the performance and efficiency of the proposed LITE framework.
