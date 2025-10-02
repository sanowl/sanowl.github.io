# AI Algorithms Comprehensive Guide by San

**Author:** San Hashimhama | AI Researcher at Cyrion Labs | Research Lead at SourceMind Labs  
**Email:** san.hashimhama@outlook.com  
**Portfolio:** [sanowl.github.io](https://sanowl.github.io)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Neural Networks & Deep Learning](#neural-networks--deep-learning)
3. [Transformers & Attention Mechanisms](#transformers--attention-mechanisms)
4. [Reinforcement Learning](#reinforcement-learning)
5. [Unsupervised Learning](#unsupervised-learning)
6. [Optimization Algorithms](#optimization-algorithms)
7. [Computer Vision](#computer-vision)
8. [Natural Language Processing](#natural-language-processing)
9. [Advanced Architectures](#advanced-architectures)
10. [References & Links](#references--links)

---

## Introduction

Welcome to my comprehensive guide to AI algorithms! As an AI researcher with over 9 years of coding experience, I've compiled detailed explanations of fundamental and advanced algorithms that power modern artificial intelligence. Each section includes mathematical formulations, visual representations, and practical insights from my research at Cyrion Labs and SourceMind Labs.

This guide reflects my journey from building my first movie review website at age 12 to researching biologically-inspired language models. Every algorithm is explained with the depth and clarity I wish I had when starting my AI journey.

---

## Neural Networks & Deep Learning

### 1. Multi-Layer Perceptron (MLP)

**Mathematical Foundation:**

For a neural network with L layers, the forward propagation is defined as:

```
z^(l) = W^(l) * a^(l-1) + b^(l)
a^(l) = σ(z^(l))
```

Where:
- `z^(l)` is the linear combination at layer l
- `W^(l)` is the weight matrix at layer l
- `a^(l)` is the activation at layer l
- `b^(l)` is the bias vector at layer l
- `σ` is the activation function

**Backpropagation Algorithm:**

The cost function gradient with respect to weights:

```
∂C/∂W^(l) = δ^(l) * (a^(l-1))^T
∂C/∂b^(l) = δ^(l)
```

Where the error term δ is computed as:

```
δ^(L) = ∇_a C ⊙ σ'(z^(L))  (output layer)
δ^(l) = ((W^(l+1))^T δ^(l+1)) ⊙ σ'(z^(l))  (hidden layers)
```

**Visual Representation:**

```
Input Layer    Hidden Layer 1    Hidden Layer 2    Output Layer
    x₁ ────────○─────────────────○─────────────────○ ŷ₁
    x₂ ────────○─────────────────○─────────────────○ ŷ₂
    x₃ ────────○─────────────────○─────────────────○ ŷ₃
    x₄ ────────○─────────────────○
               
Forward Pass: Input → Hidden → Output
Backward Pass: ∇Output ← ∇Hidden ← ∇Input
```

**Activation Functions:**

1. **Sigmoid:** `σ(x) = 1/(1 + e^(-x))`
2. **ReLU:** `σ(x) = max(0, x)`
3. **Tanh:** `σ(x) = (e^x - e^(-x))/(e^x + e^(-x))`
4. **Leaky ReLU:** `σ(x) = max(0.01x, x)`

**San's Implementation:** [Neural Network from Scratch](https://github.com/sanowl/neural-network-scratch)

---

### 2. Convolutional Neural Networks (CNNs)

**Mathematical Foundation:**

The convolution operation is defined as:

```
(f * g)(t) = ∫ f(τ)g(t - τ)dτ
```

For discrete 2D convolution:

```
S(i,j) = (I * K)(i,j) = ΣₘΣₙ I(m,n)K(i-m,j-n)
```

**CNN Architecture Components:**

1. **Convolution Layer:**
   ```
   Feature Map = ReLU(Input ⊛ Filter + Bias)
   Output Size = (Input Size - Filter Size + 2×Padding)/Stride + 1
   ```

2. **Pooling Layer (Max Pooling):**
   ```
   Output(i,j) = max{Input(si:si+f, sj:sj+f)}
   ```

3. **Fully Connected Layer:**
   ```
   Output = σ(W × Flattened_Input + b)
   ```

**Visual Architecture:**

```
Input Image (32×32×3)
         ↓
Conv Layer 1 (5×5 filters, 32 feature maps)
         ↓
Max Pool (2×2, stride 2)
         ↓
Conv Layer 2 (5×5 filters, 64 feature maps)
         ↓
Max Pool (2×2, stride 2)
         ↓
Flatten → FC Layer 1 (128 neurons) → FC Layer 2 (10 classes)
         ↓
Softmax → Probabilities
```

**Filter Visualization:**

```
3×3 Edge Detection Filter:    3×3 Gaussian Blur Filter:
[-1  -1  -1]                  [1/16  2/16  1/16]
[-1   8  -1]                  [2/16  4/16  2/16]
[-1  -1  -1]                  [1/16  2/16  1/16]
```

**San's Implementation:** [Advanced CNN Architectures](https://github.com/sanowl/advanced-cnn-architectures)

---

### 3. Recurrent Neural Networks (RNNs)

**Mathematical Foundation:**

Standard RNN equations:

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

**Long Short-Term Memory (LSTM):**

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    (forget gate)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    (input gate)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) (candidate values)
C_t = f_t * C_{t-1} + i_t * C̃_t        (cell state)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    (output gate)
h_t = o_t * tanh(C_t)                   (hidden state)
```

**LSTM Architecture Diagram:**

```
                 C_{t-1} ────────────────── C_t
                    │           ×           │
                    │           │           │
                    │     ┌─────┴─────┐     │
                    │     │    f_t    │     │
                    │     └───────────┘     │
                    │           │           │
                    └───────────×───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
         ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
         │   i_t   │       │   C̃_t   │       │   o_t   │
         └────┬────┘       └────┬────┘       └────┬────┘
              │                 │                 │
              └─────────×───────┘                 │
                        │                         │
                        └────────── + ────────────┘
                                    │
                                   h_t
```

**Gated Recurrent Unit (GRU):**

```
z_t = σ(W_z · [h_{t-1}, x_t])        (update gate)
r_t = σ(W_r · [h_{t-1}, x_t])        (reset gate)
h̃_t = tanh(W · [r_t * h_{t-1}, x_t]) (candidate activation)
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t (final activation)
```

**San's Implementation:** [RNN Language Models](https://github.com/sanowl/rnn-language-models)

---

## Transformers & Attention Mechanisms

### 4. Self-Attention Mechanism

**Mathematical Foundation:**

The attention mechanism computes weighted representations:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q (Query) = XW_Q
- K (Key) = XW_K  
- V (Value) = XW_V
- d_k = dimension of key vectors

**Multi-Head Attention:**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Positional Encoding:**

```
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

**Transformer Architecture Visualization:**

```
                    Output Probabilities
                           ↑
                    Linear & Softmax
                           ↑
                    Add & Norm
                           ↑
                    Feed Forward
                           ↑
                    Add & Norm
                           ↑
                Multi-Head Attention
                           ↑
                    Add & Norm
                           ↑
                Multi-Head Attention
                           ↑
                Input Embeddings + Positional Encoding
                           ↑
                     Input Tokens
```

**Attention Pattern Visualization:**

```
Query: "The cat sat on the mat"
       T  h  e  c  a  t  s  a  t  o  n  t  h  e  m  a  t
    T [■][□][□][□][□][■][□][□][■][□][□][□][□][□][□][□][□]
    h [□][■][■][□][□][□][□][□][□][□][□][□][■][■][□][□][□]
    e [□][■][■][□][□][□][□][□][□][□][□][□][■][■][□][□][□]
    c [□][□][□][■][■][■][□][□][□][□][□][□][□][□][□][□][□]
    a [□][□][□][■][■][■][□][■][□][□][□][□][□][□][□][■][□]
    t [■][□][□][■][■][■][□][■][■][□][□][□][□][□][□][■][■]
```

**San's Implementation:** [Transformer from Scratch](https://github.com/sanowl/transformer-architecture)

---

### 5. BERT & GPT Architectures

**BERT (Bidirectional Encoder Representations from Transformers):**

**Pre-training Objectives:**

1. **Masked Language Modeling (MLM):**
   ```
   L_MLM = -Σᵢ log P(xᵢ | x̂)
   ```

2. **Next Sentence Prediction (NSP):**
   ```
   L_NSP = -log P(IsNext | [CLS])
   ```

**GPT (Generative Pre-trained Transformer):**

**Autoregressive Language Modeling:**

```
P(x₁, x₂, ..., xₙ) = ∏ᵢ₌₁ⁿ P(xᵢ | x₁, x₂, ..., xᵢ₋₁)

L_LM = -Σᵢ log P(xᵢ | x₁, x₂, ..., xᵢ₋₁)
```

**Architecture Comparison:**

```
BERT (Encoder-only):                GPT (Decoder-only):
Input → [CLS] token₁ [SEP] token₂   Input → token₁ token₂ ... tokenₙ
         ↓                                   ↓
    Bidirectional                       Causal/Unidirectional
    Self-Attention                      Self-Attention
         ↓                                   ↓
    Classification                      Next Token Prediction
```

**San's Implementation:** [BERT Fine-tuning](https://github.com/sanowl/bert-fine-tuning) | [GPT Implementation](https://github.com/sanowl/gpt-from-scratch)

---

## Reinforcement Learning

### 6. Q-Learning Algorithm

**Mathematical Foundation:**

The Q-learning update rule:

```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

Where:
- s: current state
- a: action taken
- r: reward received
- s': next state
- α: learning rate
- γ: discount factor

**Bellman Equation:**

```
Q*(s, a) = E[r + γ max Q*(s', a') | s, a]
```

**Q-Learning Algorithm Flow:**

```
Initialize Q(s,a) arbitrarily
For each episode:
    Initialize s
    For each step of episode:
        Choose a from s using ε-greedy policy
        Take action a, observe r, s'
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        s ← s'
    Until s is terminal
```

**ε-Greedy Policy:**

```
π(s) = {
    argmax Q(s,a)  with probability 1-ε
    random action  with probability ε
}
```

**San's Implementation:** [Q-Learning Game AI](https://github.com/sanowl/q-learning-game-ai)

---

### 7. Policy Gradient Methods

**REINFORCE Algorithm:**

The policy gradient theorem:

```
∇_θ J(θ) = E_π[∇_θ log π(a|s) Q^π(s,a)]
```

**Actor-Critic Algorithm:**

```
Critic Update: w ← w + α_w δ ∇_w V(s)
Actor Update: θ ← θ + α_θ δ ∇_θ log π(a|s)

where δ = r + γV(s') - V(s)
```

**Proximal Policy Optimization (PPO):**

```
L^CLIP(θ) = E_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

**Policy Gradient Flow:**

```
Environment State (s)
        ↓
Policy Network π(a|s)
        ↓
Action Selection (a)
        ↓
Environment Interaction
        ↓
Reward (r) & Next State (s')
        ↓
Advantage Estimation A(s,a)
        ↓
Policy Update: θ ← θ + α∇_θ log π(a|s)A(s,a)
```

**San's Implementation:** [Policy Gradient RL](https://github.com/sanowl/Self-Correcting-LLM--Reinforcement-Learning-)

---

## Unsupervised Learning

### 8. K-Means Clustering

**Mathematical Foundation:**

Objective function (minimize within-cluster sum of squares):

```
J = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
```

**Algorithm Steps:**

1. Initialize k centroids μ₁, μ₂, ..., μₖ
2. Assign each point to closest centroid:
   ```
   c(i) = argmin ||xᵢ - μⱼ||²
   ```
3. Update centroids:
   ```
   μⱼ = (1/|Cⱼ|) Σₓ∈Cⱼ x
   ```
4. Repeat steps 2-3 until convergence

**Visual Representation:**

```
Iteration 0:           Iteration 1:           Iteration N:
  × × ×                  × × ×                  × × ×
× C₁  ×      →        ×  C₁'×      →        ×  C₁"×
  × × ×                  × × ×                  × × ×

    ○○○                    ○○○                    ○○○
  ○ C₂ ○                ○ C₂'○                ○ C₂"○
    ○○○                    ○○○                    ○○○
```

**Elbow Method for Optimal K:**

```
WCSS vs Number of Clusters
│
│  *
│    *
│      *
│        *___
│            *___
│                *___
└────────────────────────→
 1   2   3   4   5   6   K
```

**San's Implementation:** [Clustering Algorithms](https://github.com/sanowl/clustering-algorithms)

---

### 9. Principal Component Analysis (PCA)

**Mathematical Foundation:**

1. **Standardize data:** X_std = (X - μ)/σ
2. **Compute covariance matrix:** C = (1/n-1)X^T X
3. **Find eigenvalues and eigenvectors:** Cv = λv
4. **Sort by eigenvalue:** λ₁ ≥ λ₂ ≥ ... ≥ λₙ
5. **Transform data:** Y = XW

**Eigenvalue Decomposition:**

```
C = PΛP^T

where:
P = [v₁ v₂ ... vₙ] (eigenvectors)
Λ = diag(λ₁, λ₂, ..., λₙ) (eigenvalues)
```

**Variance Explained:**

```
Explained Variance Ratio = λᵢ / Σⱼλⱼ
```

**PCA Visualization:**

```
Original 2D Data:          After PCA:
     │                          │
   × │ ×                      × │
 ×   │   ×            →         │ ×
×    │PC1 ×                     │   ×
─────┼─────                     ├─────
×    │     ×                    │
 ×   │   ×                      │
   × │ ×                        │
     │PC2
```

**San's Implementation:** [Dimensionality Reduction](https://github.com/sanowl/dimensionality-reduction)

---

## Optimization Algorithms

### 10. Gradient Descent Variants

**Batch Gradient Descent:**

```
θ = θ - α∇_θ J(θ)
```

**Stochastic Gradient Descent (SGD):**

```
θ = θ - α∇_θ J(θ; x⁽ⁱ⁾, y⁽ⁱ⁾)
```

**Mini-batch Gradient Descent:**

```
θ = θ - α∇_θ J(θ; x⁽ⁱ:ⁱ⁺ⁿ⁾, y⁽ⁱ:ⁱ⁺ⁿ⁾)
```

**Momentum:**

```
v_t = βv_{t-1} + α∇_θ J(θ)
θ = θ - v_t
```

**AdaGrad:**

```
G_t = G_{t-1} + (∇_θ J(θ))²
θ = θ - α/(√G_t + ε) ∇_θ J(θ)
```

**Adam (Adaptive Moment Estimation):**

```
m_t = β₁m_{t-1} + (1-β₁)∇_θ J(θ)     (1st moment)
v_t = β₂v_{t-1} + (1-β₂)(∇_θ J(θ))²  (2nd moment)
m̂_t = m_t/(1-β₁^t)                   (bias correction)
v̂_t = v_t/(1-β₂^t)                   (bias correction)
θ = θ - α m̂_t/(√v̂_t + ε)
```

**Convergence Comparison:**

```
Loss
│  \
│   \  SGD
│    \
│     \ ···
│      \   \  Momentum
│       \    \
│        \    \.
│         \     \..
│          \       \.. Adam
│           \         \....
└────────────────────────────→ Iterations
```

**San's Implementation:** [Optimization Methods](https://github.com/sanowl/optimization-algorithms)

---

## Computer Vision

### 11. Object Detection - YOLO Algorithm

**Mathematical Foundation:**

YOLO divides image into S×S grid and predicts B bounding boxes per cell:

**Bounding Box Prediction:**
```
Box = (x, y, w, h, confidence)
x, y = center coordinates relative to cell
w, h = width, height relative to image
confidence = P(Object) × IOU
```

**Loss Function:**

```
L = λ_coord Σᵢ₌₀^{S²} Σⱼ₌₀^B 𝟙ᵢⱼᵒᵇʲ [(xᵢ - x̂ᵢ)² + (yᵢ - ŷᵢ)²]
    + λ_coord Σᵢ₌₀^{S²} Σⱼ₌₀^B 𝟙ᵢⱼᵒᵇʲ [(√wᵢ - √ŵᵢ)² + (√hᵢ - √ĥᵢ)²]
    + Σᵢ₌₀^{S²} Σⱼ₌₀^B 𝟙ᵢⱼᵒᵇʲ (Cᵢ - Ĉᵢ)²
    + λ_noobj Σᵢ₌₀^{S²} Σⱼ₌₀^B 𝟙ᵢⱼⁿᵒᵒᵇʲ (Cᵢ - Ĉᵢ)²
    + Σᵢ₌₀^{S²} 𝟙ᵢᵒᵇʲ Σ_{c∈classes} (pᵢ(c) - p̂ᵢ(c))²
```

**YOLO Architecture:**

```
Input Image (448×448×3)
        ↓
24 Convolutional Layers
        ↓
2 Fully Connected Layers
        ↓
Output (7×7×30)
        ↓
Reshape → (7, 7, 2 boxes + 20 classes)
```

**Non-Maximum Suppression (NMS):**

```
1. Sort boxes by confidence score
2. While boxes remain:
   - Pick box with highest confidence
   - Remove boxes with IoU > threshold
   - Add picked box to final detections
```

**Intersection over Union (IoU):**

```
IoU = Area of Intersection / Area of Union
    = |A ∩ B| / |A ∪ B|
```

**San's Implementation:** [YOLO Object Detection](https://github.com/sanowl/yolo-object-detection)

---

### 12. Generative Adversarial Networks (GANs)

**Mathematical Foundation:**

**Minimax Game:**

```
min_G max_D V(D,G) = E_{x~p_data(x)}[log D(x)] + E_{z~p_z(z)}[log(1-D(G(z)))]
```

**Training Algorithm:**

```
For number of training iterations:
    For k steps:
        Sample m noise samples {z⁽¹⁾,...,z⁽ᵐ⁾} from p_z(z)
        Sample m examples {x⁽¹⁾,...,x⁽ᵐ⁾} from p_data(x)
        Update discriminator by ascending:
        ∇_θd (1/m) Σᵢ[log D(x⁽ⁱ⁾) + log(1-D(G(z⁽ⁱ⁾)))]
    
    Sample m noise samples {z⁽¹⁾,...,z⁽ᵐ⁾} from p_z(z)
    Update generator by descending:
    ∇_θg (1/m) Σᵢ log(1-D(G(z⁽ⁱ⁾)))
```

**GAN Architecture:**

```
Random Noise z ~ N(0,1)
        ↓
Generator Network G(z)
        ↓
Fake Image G(z)
        ↓
Discriminator D(x) ← Real Image x
        ↓
Probability [0,1]
        ↓
Real/Fake Classification
```

**Loss Functions:**

**Generator Loss:**
```
L_G = -E_{z~p_z}[log D(G(z))]
```

**Discriminator Loss:**
```
L_D = -E_{x~p_data}[log D(x)] - E_{z~p_z}[log(1-D(G(z)))]
```

**Popular GAN Variants:**

1. **DCGAN:** Deep Convolutional GAN
2. **WGAN:** Wasserstein GAN with Earth-Mover distance
3. **StyleGAN:** Style-based Generator Architecture
4. **CycleGAN:** Unpaired Image-to-Image Translation

**San's Implementation:** [GAN Implementations](https://github.com/sanowl/gan-implementations)

---

## Natural Language Processing

### 13. Word2Vec Algorithm

**Mathematical Foundation:**

**Skip-gram Model:**

Maximize the log-likelihood:

```
L = (1/T) Σᵢ₌₁ᵀ Σⱼ∈C(i) log P(wⱼ|wᵢ)
```

Where P(wⱼ|wᵢ) is defined using softmax:

```
P(wⱼ|wᵢ) = exp(vʷʲᵀ vʷⁱ) / Σₖ₌₁ᵂ exp(vʷₖᵀ vʷⁱ)
```

**Negative Sampling:**

Instead of computing expensive softmax, sample k negative words:

```
L = log σ(vʷᵒᵀ vʷᵢ) + Σⱼ₌₁ᵏ log σ(-vʷⱼᵀ vʷᵢ)
```

**CBOW (Continuous Bag of Words):**

Predict center word from context:

```
P(wᵢ|Context) = exp(vʷⁱᵀ (Σⱼ∈C(i) vʷⱼ)/|C(i)|) / Σₖ₌₁ᵂ exp(vʷₖᵀ (Σⱼ∈C(i) vʷⱼ)/|C(i)|)
```

**Word Vector Space Visualization:**

```
         king
          │
          │
    man───┼───woman
          │
          │
        queen

Vector Operations:
king - man + woman ≈ queen
```

**San's Implementation:** [Word2Vec from Scratch](https://github.com/sanowl/word2vec-implementation)

---

### 14. Sequence-to-Sequence Models

**Mathematical Foundation:**

**Encoder-Decoder Architecture:**

```
Encoder: h_t = f(x_t, h_{t-1})
Context: c = q({h_1, h_2, ..., h_T})
Decoder: s_t = f(y_{t-1}, s_{t-1}, c)
Output: p(y_t|y_1,...,y_{t-1}, x) = g(y_{t-1}, s_t, c)
```

**Attention Mechanism:**

```
α_{t,i} = exp(e_{t,i}) / Σₖ₌₁ᵀ exp(e_{t,k})
e_{t,i} = a(s_{t-1}, h_i)
c_t = Σᵢ₌₁ᵀ α_{t,i} h_i
```

**Seq2Seq with Attention:**

```
Input:  [SOS] Je suis étudiant [EOS]
         │    │   │     │       │
Encoder: h₁   h₂  h₃    h₄      h₅
                                │
                        Context Vector
                                │
Decoder: s₁   s₂  s₃    s₄      s₅
         │    │   │     │       │
Output: [SOS] I  am   student  [EOS]
```

**Beam Search Decoding:**

```
At each step, keep top k sequences:

Step 1: [I: 0.7, We: 0.2, They: 0.1]
Step 2: [I am: 0.6, I was: 0.5, We are: 0.15]
Step 3: [I am happy: 0.5, I was there: 0.4, I am sad: 0.3]
```

**San's Implementation:** [Seq2Seq Models](https://github.com/sanowl/seq2seq-models)

---

## Advanced Architectures

### 15. Vision Transformers (ViTs)

**Mathematical Foundation:**

**Image Patch Embedding:**

```
x_p = [x₁; x₂; ...; x_N] ∈ ℝ^{N×(P²·C)}
```

Where:
- N = HW/P² (number of patches)
- P = patch size
- C = number of channels

**Position Embedding:**

```
z₀ = [x_class; x₁E; x₂E; ...; x_NE] + E_pos
```

**Transformer Encoder:**

```
z'_l = MSA(LN(z_{l-1})) + z_{l-1}
z_l = MLP(LN(z'_l)) + z'_l
```

**ViT Architecture:**

```
Input Image (224×224×3)
        ↓
Patch Embedding (16×16 patches)
        ↓
Linear Projection + Position Embedding
        ↓
Transformer Encoder × L layers
        ↓
Classification Head ([CLS] token)
        ↓
Output Classes
```

**Patch Visualization:**

```
Original Image:          Patches (16×16):
┌─────────────────┐     ┌──┬──┬──┬──┬──┐
│                 │     │01│02│03│04│05│
│     Object      │ →   ├──┼──┼──┼──┼──┤
│   Recognition   │     │06│07│08│09│10│
│                 │     ├──┼──┼──┼──┼──┤
└─────────────────┘     │11│12│13│14│15│
                        └──┴──┴──┴──┴──┘
```

**San's Implementation:** [Vision Transformers](https://github.com/sanowl/vision-transformers)

---

### 16. Diffusion Models

**Mathematical Foundation:**

**Forward Diffusion Process:**

```
q(x_{1:T}|x_0) = ∏ᵗ₌₁ᵀ q(x_t|x_{t-1})
q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
```

**Reverse Diffusion Process:**

```
p_θ(x_{0:T}) = p(x_T) ∏ᵗ₌₁ᵀ p_θ(x_{t-1}|x_t)
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))
```

**Training Objective (Denoising Score Matching):**

```
L_simple = E_{t,x_0,ε}[||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t)ε, t)||²]
```

**DDPM Algorithm:**

```
Training:
1. Sample x_0 ~ q(x_0)
2. Sample t ~ Uniform({1,...,T})
3. Sample ε ~ N(0,I)
4. Take gradient step on ∇_θ||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t)ε, t)||²

Sampling:
1. x_T ~ N(0,I)
2. For t = T,...,1:
   x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t,t)) + σ_t z
   where z ~ N(0,I) if t > 1, else z = 0
```

**Diffusion Process Visualization:**

```
Original Image → Add Noise → Add More Noise → Pure Noise
     x_0     →     x_1    →      x_2      →     x_T
      ↑                                         ↓
Denoise   ←    Denoise    ←   Denoise    ←   Denoise
```

**San's Implementation:** [Diffusion Models](https://github.com/sanowl/diffusion-models)

---

### 17. Graph Neural Networks (GNNs)

**Mathematical Foundation:**

**Message Passing Framework:**

```
m_ij^(l) = Message(h_i^(l), h_j^(l), e_ij)
m_i^(l) = Aggregate({m_ij^(l) : j ∈ N(i)})
h_i^(l+1) = Update(h_i^(l), m_i^(l))
```

**Graph Convolutional Networks (GCN):**

```
H^(l+1) = σ(D̃^(-1/2) ÃD̃^(-1/2) H^(l) W^(l))
```

Where:
- Ã = A + I (adjacency matrix with self-loops)
- D̃ = degree matrix of Ã

**GraphSAGE:**

```
h_N(v)^(l) = AGGREGATE_l({h_u^(l-1), ∀u ∈ N(v)})
h_v^(l) = σ(W^(l) · CONCAT(h_v^(l-1), h_N(v)^(l)))
```

**Graph Attention Networks (GAT):**

```
α_ij = exp(LeakyReLU(a^T[Wh_i||Wh_j])) / Σₖ∈N(i) exp(LeakyReLU(a^T[Wh_i||Wh_k]))
h_i' = σ(Σⱼ∈N(i) α_ij Wh_j)
```

**Graph Structure Example:**

```
Node Features:    Adjacency Matrix:      After GCN:
A: [1,0,1]       A B C D E              A': [0.8,0.2,0.9]
B: [0,1,0]    A [0 1 1 0 0]             B': [0.3,0.7,0.1]
C: [1,1,0]    B [1 0 1 1 0]             C': [0.6,0.5,0.4]
D: [0,0,1]    C [1 1 0 1 1]             D': [0.4,0.3,0.8]
E: [1,0,0]    D [0 1 1 0 1]             E': [0.7,0.1,0.6]
              E [0 0 1 1 0]
```

**San's Implementation:** [Graph Neural Networks](https://github.com/sanowl/graph-neural-networks)

---

### 18. Meta-Learning Algorithms

**Mathematical Foundation:**

**Model-Agnostic Meta-Learning (MAML):**

```
θ' = θ - α∇_θ L_{T_i}(f_θ)  (task-specific adaptation)
θ ← θ - β∇_θ Σᵢ L_{T_i}(f_{θ'})  (meta-update)
```

**Prototypical Networks:**

```
c_k = (1/|S_k|) Σ_{(x_i,y_i)∈S_k} f_φ(x_i)  (prototype for class k)
p(y=k|x) = exp(-d(f_φ(x), c_k)) / Σₖ' exp(-d(f_φ(x), c_k'))
```

**Matching Networks:**

```
ŷ = Σᵢ₌₁ᵏ a(f(x̂), g(x_i)) y_i
a(x̂, x_i) = exp(c(f(x̂), g(x_i))) / Σⱼ₌₁ᵏ exp(c(f(x̂), g(x_j)))
```

**Meta-Learning Process:**

```
Meta-Training:
Tasks: T₁, T₂, ..., Tₙ
For each task:
  1. Sample support set S
  2. Adapt: θ' = Adapt(θ, S)
  3. Evaluate on query set Q
  4. Update: θ ← MetaUpdate(θ, L(θ', Q))

Meta-Testing:
New task T*:
  1. Sample support set S*
  2. Adapt: θ* = Adapt(θ, S*)
  3. Classify query examples
```

**Few-Shot Learning Visualization:**

```
Support Set (K-shot, N-way):
Class A: [img1, img2, ..., imgK]
Class B: [img1, img2, ..., imgK]
Class C: [img1, img2, ..., imgK]

Query Set:
Test image → Classifier → Predicted Class
```

**San's Implementation:** [Meta-Learning Algorithms](https://github.com/sanowl/meta-learning-algorithms)

---

### 19. Federated Learning

**Mathematical Foundation:**

**FedAvg Algorithm:**

```
Server Update:
w_{t+1} = Σₖ₌₁ᴷ (n_k/n) w_k^{t+1}

Client k Update:
w_k^{t+1} = LocalUpdate(k, w_t)

LocalUpdate(k, w):
B ← split P_k into batches of size B
for each local epoch e from 1 to E:
    for batch b in B:
        w ← w - η∇l(w; b)
return w
```

**Differential Privacy in FL:**

```
w̃_k = w_k + N(0, σ²I)  (add Gaussian noise)
Privacy Budget: ε = q²T / (2σ²)
```

**Federated Learning Architecture:**

```
                Central Server
                     │
      ┌──────────────┼──────────────┐
      │              │              │
   Client 1      Client 2      Client 3
   [Local Data]  [Local Data]  [Local Data]
      │              │              │
   Local Model   Local Model   Local Model
      │              │              │
      └──────────────┼──────────────┘
                     │
              Aggregate Updates
                     │
              Global Model Update
```

**Communication Efficiency:**

```
Compression Techniques:
1. Quantization: w ← Quantize(w, b bits)
2. Sparsification: w ← TopK(w, k%)
3. Low-rank approximation: W ≈ UV^T
```

**San's Implementation:** [Federated Learning Framework](https://github.com/sanowl/federated-learning)

---

### 20. Neuromorphic Computing & Spiking Neural Networks

**Mathematical Foundation:**

**Leaky Integrate-and-Fire (LIF) Neuron:**

```
τ_m dV/dt = -(V - V_rest) + I(t)

If V ≥ V_th:
    V → V_reset
    emit spike
```

**Spike-Timing-Dependent Plasticity (STDP):**

```
Δw = {
    A₊ exp(-(t_post - t_pre)/τ₊)  if t_post > t_pre
    -A₋ exp(-(t_pre - t_post)/τ₋)  if t_pre > t_post
}
```

**Spiking Neural Network Architecture:**

```
Input Spikes → Encoding Layer → Hidden Layers → Output Layer
     │              │               │              │
Poisson Rate     LIF Neurons    LIF Neurons   Rate Decoding
   Coding
```

**Spike Train Representation:**

```
Time: 0  10  20  30  40  50  60  70  80  90  ms
Neuron 1: |     |        |     |         |
Neuron 2:    |     |  |        |     |
Neuron 3: |  |        |     |        |  |
```

**Energy Efficiency Comparison:**

```
Power Consumption:
Von Neumann: ████████████████████ 100W
GPU:         ████████████ 250W
Neuromorphic: ██ 20mW
Human Brain:  ████ 20W
```

**San's Implementation:** [Spiking Neural Networks](https://github.com/sanowl/spiking-neural-networks)

---

## References & Links

### San's Research Projects & Implementations

**Core Projects:**
- [LSLM - Listening-while-Speaking Language Model](https://github.com/sanowl/LSLM-Listening-while-Speaking-Language-Model)
- [Self-Correcting LLM with Reinforcement Learning](https://github.com/sanowl/Self-Correcting-LLM--Reinforcement-Learning-)
- [OmegaPRM - Process Supervision Framework](https://github.com/sanowl/OmegaPRM)
- [CoRAG - Chain-of-Retrieval Augmented Generation](https://github.com/sanowl/CoRAG)
- [Drag-and-Drop LLMs](https://github.com/sanowl/Drag-and-Drop-LLMs-Zero-Shot-Prompt-to-Weights)

**Algorithm Implementations:**
- [Transformer Architecture from Scratch](https://github.com/sanowl/transformer-architecture)
- [Neural Network Fundamentals](https://github.com/sanowl/neural-network-scratch)
- [Advanced CNN Architectures](https://github.com/sanowl/advanced-cnn-architectures)
- [RNN Language Models](https://github.com/sanowl/rnn-language-models)
- [BERT Fine-tuning](https://github.com/sanowl/bert-fine-tuning)
- [GPT Implementation](https://github.com/sanowl/gpt-from-scratch)
- [Q-Learning Game AI](https://github.com/sanowl/q-learning-game-ai)
- [Clustering Algorithms](https://github.com/sanowl/clustering-algorithms)
- [Dimensionality Reduction](https://github.com/sanowl/dimensionality-reduction)
- [Optimization Methods](https://github.com/sanowl/optimization-algorithms)
- [YOLO Object Detection](https://github.com/sanowl/yolo-object-detection)
- [GAN Implementations](https://github.com/sanowl/gan-implementations)
- [Word2Vec Implementation](https://github.com/sanowl/word2vec-implementation)
- [Seq2Seq Models](https://github.com/sanowl/seq2seq-models)
- [Vision Transformers](https://github.com/sanowl/vision-transformers)
- [Diffusion Models](https://github.com/sanowl/diffusion-models)
- [Graph Neural Networks](https://github.com/sanowl/graph-neural-networks)
- [Meta-Learning Algorithms](https://github.com/sanowl/meta-learning-algorithms)
- [Federated Learning Framework](https://github.com/sanowl/federated-learning)
- [Spiking Neural Networks](https://github.com/sanowl/spiking-neural-networks)

### Mathematical Resources

**Linear Algebra & Calculus:**
- Gilbert Strang's Linear Algebra Course (MIT OpenCourseWare)
- Paul's Online Math Notes (Differential Equations)
- Khan Academy (Multivariate Calculus)

**Probability & Statistics:**
- Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani)
- Pattern Recognition and Machine Learning (Christopher Bishop)
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)

**Deep Learning:**
- Deep Learning Book (Ian Goodfellow, Yoshua Bengio, Aaron Courville)
- Neural Networks and Deep Learning (Michael Nielsen)
- Hands-On Machine Learning (Aurélien Géron)

### Academic Papers & Research

**Foundational Papers:**
- Attention Is All You Need (Vaswani et al., 2017)
- BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
- Language Models are Few-Shot Learners (Brown et al., 2020)
- Generative Adversarial Networks (Goodfellow et al., 2014)
- Deep Residual Learning for Image Recognition (He et al., 2016)

**Recent Advances:**
- LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
- PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)
- Flamingo: Few-shot Learning in Vision-Language Models (Alayrac et al., 2022)

### Contact & Collaboration

**San Hashimhama**
- **Email:** san.hashimhama@outlook.com
- **GitHub:** [@sanowl](https://github.com/sanowl)
- **Research Positions:** 
  - Research Lead at SourceMind Labs
  - AI Researcher at Cyrion Labs
- **Education:** AUIS University (Current)

**Specializations:**
- Biologically-inspired language models
- Transformer architectures and attention mechanisms
- Reinforcement learning for language models
- Process supervision and mathematical reasoning
- Retrieval-augmented generation systems

---

## Conclusion

This comprehensive guide represents over 9 years of AI research and development, distilled into mathematical foundations, practical implementations, and visual explanations. From my early days building movie review websites to current research on biologically-inspired language models at Cyrion Labs and SourceMind Labs, each algorithm has been carefully explained with the depth necessary for both understanding and implementation.

The field of AI continues to evolve rapidly, but these fundamental algorithms remain the building blocks of more complex systems. Whether you're starting your AI journey or looking to deepen your understanding, I hope this guide serves as both a reference and inspiration for your own research and development.

**"The best way to understand intelligence is to build it."** - San Hashimhama

---

*© 2025 San Hashimhama. This guide is open-source and available for educational use. Please cite appropriately when using in academic work.*