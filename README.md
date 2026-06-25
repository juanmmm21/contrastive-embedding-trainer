# contrastive-embedding-trainer

Pipeline personalizado en PyTorch para entrenar embeddings de texto utilizando una arquitectura de red siamesa y Triplet Cosine Loss.

Este proyecto permite realizar el ajuste fino (fine-tuning) de codificadores basados en Transformer para tareas de busqueda semantica, agrupamiento (clustering) y sistemas de recuperacion de informacion (RAG).

## Arquitectura y Fundamentos Teoricos

### 1. Red Siamesa (Siamese Network)
El sistema emplea un unico codificador Transformer base (por defecto `distilbert-base-uncased`) cuyos pesos se comparten en una estructura siamesa. Durante el entrenamiento, procesamos en paralelo tres textos (el triplete):
*   **Anchor (Ancla):** El texto de referencia.
*   **Positive (Positivo):** Un texto semanticamente similar o relacionado al Anchor.
*   **Negative (Negativo):** Un texto semanticamente disimil o no relacionado al Anchor.

Cada texto genera su representacion densa a traves del mismo codificador.

### 2. Mean Pooling sobre Estados Ocultos
En lugar de depender unicamente del token especial `[CLS]` (que a menudo sufre de cuellos de botella informacionales en tareas de similitud de oraciones), este pipeline realiza un promedio aritmetico (Mean Pooling) de los estados ocultos de la ultima capa del Transformer para todos los tokens de la oracion, ponderado y filtrado mediante la mascara de atencion (`attention_mask`) para ignorar los tokens de padding.

Matematicamente, para una secuencia de embeddings de tokens $H = \{h_1, h_2, ..., h_n\}$ y una mascara de atencion binaria $M = \{m_1, m_2, ..., m_n\}$:

$$\text{Embedding} = \frac{\sum_{i=1}^{n} h_i \cdot m_i}{\sum_{i=1}^{n} m_i}$$

Finalmente, se aplica una normalizacion L2 al vector resultante para simplificar la busqueda vectorial (el producto escalar directo equivale a la similitud de coseno).

### 3. Triplet Cosine Loss
La funcion de perdida optimiza el espacio vectorial forzando a que la distancia de coseno entre el Anchor y el Positive sea menor que la distancia entre el Anchor y el Negative por un margen establecido.

La distancia de coseno se define a partir de la similitud de coseno:

$$D_{cos}(x, y) = 1.0 - \text{CosineSimilarity}(x, y)$$

La formula de la perdida para un triplete es:

$$\mathcal{L} = \max(D_{cos}(Anchor, Positive) - D_{cos}(Anchor, Negative) + m, 0)$$

donde $m$ es el margen de penalizacion (por defecto `0.3`).

## Conexion con el Ecosistema

Este proyecto interactua con otros modulos de la infraestructura `ai-core-infra`:
*   **semantic-chunking-engine:** Los fragmentos semanticos generados por dicho modulo se pueden vectorizar utilizando el modelo entrenado aqui.
*   **nano-vector-db:** Los embeddings entrenados y exportados por este pipeline se indexan directamente en la base de datos vectorial para busquedas de vecinos mas cercanos (K-NN).
*   **hybrid-search-retrieval-pipeline:** Provee los embeddings densos requeridos para la rama de recuperacion vectorial.

*Nota de diseño:* Aunque este ecosistema incluye el proyecto `bpe-tokenizer-from-scratch`, los modelos de Hugging Face pre-entrenados como DistilBERT estan fuertemente acoplados a su vocabulario original y tokenizador WordPiece. Forzar un tokenizador personalizado alteraria la indexacion de los pesos cargados, por lo que este modulo utiliza el tokenizador oficial de la arquitectura base para garantizar la precision y transferencia de aprendizaje.

## Estructura del Proyecto

*   **model.py:** Define la red siamesa `SiameseTransformer` y la clase `TripletCosineLoss`.
*   **dataset.py:** Define el `TripletDataset` para tokenizacion perezosa en PyTorch.
*   **trainer.py:** Implementa el bucle de entrenamiento, optimizacion AdamW, validacion y exportacion.
*   **example.py:** Demostracion interactiva de ajuste fino rapido sobre un dataset de tripletes ficticios.
*   **test_trainer.py:** Conjunto de pruebas unitarias locales.

## Instalacion y Requisitos

1. Asegurate de contar con Python 3.10 o superior.
2. Crea e inicia un entorno virtual dentro de la carpeta del proyecto:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Instala las dependencias especificadas en requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

## Instrucciones de Uso

### Ejecutar Pruebas Unitarias
Para validar de forma aislada e independiente de la conexion de red todos los componentes principales:
```bash
python -m unittest test_trainer.py
```

### Ejecutar Demostracion Interactiva
Para iniciar un ciclo completo de entrenamiento rapido con tripletes demostrativos de diferentes dominios (cocina, astronomia y programacion) y comparar las similitudes de coseno antes y despues del entrenamiento:
```bash
python example.py
```
El script descargara una vez el modelo base (DistilBERT), ejecutara el bucle por varias epocas en el dispositivo mas veloz disponible (GPU, MPS o CPU) y guardara el modelo ajustado y su tokenizador en el directorio `model_output/`.
