# Contrastive Embedding Trainer

Custom PyTorch pipeline for training text embeddings using a siamese network architecture and Triplet Cosine Loss. This project enables fine-tuning of Transformer-based encoders for semantic search, clustering, and retrieval-augmented generation (RAG) tasks.

## Architecture and Theoretical Foundations

The system uses a single base Transformer encoder (default `distilbert-base-uncased`) whose weights are shared in a siamese structure. During training, we process three texts in parallel (the triplet):
*   **Anchor:** The reference text.
*   **Positive:** A text that is semantically similar or related to the Anchor.
*   **Negative:** A text that is semantically dissimilar or unrelated to the Anchor.

Each text generates its dense representation through the same encoder.

### 1. Mean Pooling over Hidden States
Instead of relying solely on the special `[CLS]` token (which often suffers from information bottlenecks in sentence-similarity tasks), this pipeline performs an arithmetic average (Mean Pooling) of the Transformer's last-layer hidden states across all tokens in the sentence, weighted and filtered using the attention mask (`attention_mask`) to ignore padding tokens.

Mathematically, for a sequence of token embeddings $H = \{h_1, h_2, ..., h_n\}$ and a binary attention mask $M = \{m_1, m_2, ..., m_n\}$:

$$\text{Embedding} = \frac{\sum_{i=1}^{n} h_i \cdot m_i}{\sum_{i=1}^{n} m_i}$$

Finally, $L_2$ normalization is applied to the resulting vector to simplify cosine similarity to a dot product:

$$E_{\text{norm}} = \frac{\text{Embedding}}{\|\text{Embedding}\|_2}$$

### 2. Triplet Cosine Loss
The loss function optimizes the vector space by forcing the cosine distance between the Anchor ($A$) and the Positive ($P$) to be smaller than the distance between the Anchor and the Negative ($N$) by an established margin.

Cosine distance is defined from cosine similarity:

$$D_{\cos}(x, y) = 1.0 - \frac{x \cdot y}{\|x\|_2 \|y\|_2}$$

The loss formula for a triplet is:

$$\mathcal{L} = \max(D_{\cos}(A, P) - D_{\cos}(A, N) + m, 0)$$

where $m$ is the penalty margin (default `0.3`). Optimization seeks to minimize $\mathcal{L}$ by reducing $D_{\cos}(A, P)$ and increasing $D_{\cos}(A, N)$.

### 3. AdamW Optimization with Regularization
Training is performed using the AdamW optimizer, which decouples weight decay from the first- and second-moment gradient updates:

$$\theta_{t+1} = \theta_t - \eta_t (\hat{g}_t + \lambda \theta_t)$$

Where $\eta_t$ is the learning rate, $\lambda$ is the weight decay coefficient, and $\hat{g}_t$ is the bias-corrected gradient.

## Export Directory Structure (`model_output/`)

After fine-tuning completes, the model is exported in a format compatible with Hugging Face Transformers:
*   `pytorch_model.bin` or `model.safetensors`: The fine-tuned encoder's weights.
*   `config.json`: Structural metadata of the base model.
*   `vocab.txt` or `tokenizer.json`: The original vocabulary.
*   `tokenizer_config.json`: WordPiece tokenizer configuration.

## Using it as a package

Since `0.1.0` the code lives in the `contrastive_embedding_trainer` package and the repository is
pip-installable, so other projects can depend on it instead of copying files
or editing `sys.path`:

```bash
pip install "contrastive-embedding-trainer @ git+https://github.com/juanmmm21/contrastive-embedding-trainer.git@v0.1.0"
```

```python
from contrastive_embedding_trainer import EmbeddingTrainer, SiameseTransformer
```

The old top-level modules are kept as thin re-exports, so sibling
`ai-core-infra` repos that add this folder to `sys.path` keep working
unchanged.

## Installation Requirements

*   Python 3.10 or higher
*   PyTorch
*   Transformers (Hugging Face)
*   Numpy

To install the specified dependencies, run:
```bash
pip install -r requirements.txt
```

## Execution and Verification Guide

### 1. Run Unit Tests
Verifies random weight initialization, pooling layer dimensions, and gradient flow:
```bash
python3 -m unittest test_trainer.py
```

### 2. Run the Demo Fine-Tuning
```bash
python3 example.py
```
The script will download the base model, run 4 fine-tuning epochs, and compare the resulting cosine similarity before and after training for specific triplets (cooking, astronomy, and programming).

## Connectivity within the ai-core-infra Ecosystem

*   **semantic-chunking-engine:** The generated semantic chunks can be vectorized using the siamese model trained here.
*   **nano-vector-db:** The output embeddings are indexed directly into the HNSW index for K-nearest-neighbor (K-NN) searches.
*   **hybrid-search-retrieval-pipeline:** Provides the high-fidelity embeddings required for the dense branch.
