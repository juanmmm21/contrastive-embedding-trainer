# Contrastive Embedding Trainer

Pipeline personalizado en PyTorch para entrenar embeddings de texto utilizando una arquitectura de red siamesa y Triplet Cosine Loss. Este proyecto permite realizar el ajuste fino (fine-tuning) de codificadores basados en Transformer para tareas de busqueda semantica, agrupamiento (clustering) y sistemas de recuperacion de informacion (RAG).

## Arquitectura y Fundamentos Teoricos

El sistema emplea un unico codificador Transformer base (por defecto `distilbert-base-uncased`) cuyos pesos se comparten en una estructura siamesa. Durante el entrenamiento, procesamos en paralelo tres textos (el triplete):
*   **Anchor (Ancla):** El texto de referencia.
*   **Positive (Positivo):** Un texto semanticamente similar o relacionado al Anchor.
*   **Negative (Negativo):** Un texto semanticamente disimil o no relacionado al Anchor.

Cada texto genera su representacion densa a traves del mismo codificador.

### 1. Mean Pooling sobre Estados Ocultos
En lugar de depender unicamente del token especial `[CLS]` (que a menudo sufre de cuellos de botella informacionales en tareas de similitud de oraciones), este pipeline realiza un promedio aritmetico (Mean Pooling) de los estados ocultos de la ultima capa del Transformer para todos los tokens de la oracion, ponderado y filtrado mediante la mascara de atencion (`attention_mask`) para ignorar los tokens de padding.

Matematicamente, para una secuencia de embeddings de tokens $H = \{h_1, h_2, ..., h_n\}$ y una mascara de atencion binaria $M = \{m_1, m_2, ..., m_n\}$:

$$\text{Embedding} = \frac{\sum_{i=1}^{n} h_i \cdot m_i}{\sum_{i=1}^{n} m_i}$$

Finalmente, se aplica una normalizacion $L_2$ al vector resultante para simplificar la similitud de coseno a un producto punto:

$$E_{\text{norm}} = \frac{\text{Embedding}}{\|\text{Embedding}\|_2}$$

### 2. Triplet Cosine Loss
La funcion de perdida optimiza el espacio vectorial forzando a que la distancia de coseno entre el Anchor ($A$) y el Positive ($P$) sea menor que la distancia entre el Anchor y el Negative ($N$) por un margen establecido.

La distancia de coseno se define a partir de la similitud de coseno:

$$D_{\cos}(x, y) = 1.0 - \frac{x \cdot y}{\|x\|_2 \|y\|_2}$$

La formula de la perdida para un triplete es:

$$\mathcal{L} = \max(D_{\cos}(A, P) - D_{\cos}(A, N) + m, 0)$$

donde $m$ es el margen de penalizacion (por defecto `0.3`). La optimizacion busca minimizar $\mathcal{L}$ reduciendo $D_{\cos}(A, P)$ e incrementando $D_{\cos}(A, N)$.

### 3. Optimizacion AdamW con Regularizacion
El entrenamiento se realiza utilizando el optimizador AdamW que desacopla la penalizacion de pesos (weight decay) de las actualizaciones de gradiente de primer y segundo momento:

$$\theta_{t+1} = \theta_t - \eta_t (\hat{g}_t + \lambda \theta_t)$$

Donde $\eta_t$ es la tasa de aprendizaje, $\lambda$ es el coeficiente de weight decay, y $\hat{g}_t$ es el gradiente corregido por el sesgo.

## Estructura del Directorio de Exportacion (`model_output/`)

Tras completar el ajuste fino, el modelo se exporta en un formato compatible con Hugging Face Transformers:
*   `pytorch_model.bin` o `model.safetensors`: Los pesos del codificador ajustado.
*   `config.json`: Metadatos estructurales del modelo base.
*   `vocab.txt` o `tokenizer.json`: El vocabulario original.
*   `tokenizer_config.json`: Configuracion del tokenizador WordPiece.

## Requisitos de Instalacion

*   Python 3.10 o superior
*   PyTorch
*   Transformers (Hugging Face)
*   Numpy

Para instalar las dependencias especificadas, ejecute:
```bash
pip install -r requirements.txt
```

## Guia de Ejecucion y Verificacion

### 1. Ejecutar Pruebas Unitarias
Verifica la inicialización de pesos aleatorios, las dimensiones de la capa de pooling y la deducción de gradientes:
```bash
python3 -m unittest test_trainer.py
```

### 2. Ejecutar Ajuste Fino Demostrativo
```bash
python3 example.py
```
El script descargara el modelo base, ejecutara 4 epocas de ajuste fino y comparara la similitud de coseno resultante antes y despues del entrenamiento para tripletes especificos (cocina, astronomia y programacion).

## Conectividad en el Ecosistema ai-core-infra

*   **semantic-chunking-engine:** Los fragmentos semanticos generados se pueden vectorizar utilizando el modelo siames entrenado aqui.
*   **nano-vector-db:** Los embeddings de salida se indexan directamente en el indice HNSW para busquedas de vecinos mas cercanos (K-NN).
*   **hybrid-search-retrieval-pipeline:** Provee los embeddings de alta fidelidad requeridos para la rama densa.
