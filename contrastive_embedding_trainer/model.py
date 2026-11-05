import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class SiameseTransformer(nn.Module):
    """
    Arquitectura de Red Siamesa basada en un Transformer base (por defecto DistilBERT)
    para la generacion de representaciones vectoriales densas (embeddings) de texto.
    
    Implementa Mean Pooling sobre los estados ocultos para representar de forma semantica
    la oracion completa, una tecnica estandar de produccion superior al uso de la representacion CLS.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        super().__init__()
        # Cargamos el modelo pre-entrenado base desde Hugging Face
        self.transformer = AutoModel.from_pretrained(model_name)
        
    def mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Realiza la operacion de Mean Pooling sobre el estado oculto del Transformer,
        ignorando los tokens de padding correspondientes mediante el uso de la mascara de atencion.
        
        Args:
            model_output: Salida bruta del modelo de Hugging Face.
            attention_mask: Mascara de atencion que indica que tokens son reales (1) y cuales padding (0).
            
        Returns:
            Tensor de embeddings unificado para cada oracion.
        """
        # Obtenemos las representaciones de la ultima capa (batch_size, sequence_length, hidden_dim)
        token_embeddings = model_output.last_hidden_state
        
        # Expandimos la mascara de atencion para que coincida con las dimensiones de los embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Multiplicamos los embeddings por la mascara para silenciar los tokens de padding
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Calculamos la suma de la mascara (numero de tokens reales por secuencia)
        # Evitamos division por cero sumando un epsilon
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Promedio aritmetico de los tokens validos
        return sum_embeddings / sum_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Paso forward de una de las ramas siamesas. 
        Devuelve el embedding normalizado (L2) para un lote de texto tokenizado.
        """
        model_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(model_output, attention_mask)
        
        # Normalizamos L2 los vectores de embeddings finales para que la similitud de coseno
        # equivalga de forma directa al producto escalar de los tensores.
        return F.normalize(embeddings, p=2, dim=1)


class TripletCosineLoss(nn.Module):
    """
    Funcion de perdida de Triplete (Triplet Loss) basada en similitud y distancia del coseno.
    
    Optimiza el espacio vectorial obligando a que la distancia entre el Anchor y el Positive (similares)
    sea menor que la distancia entre el Anchor y el Negative (disimiles) por una diferencia de al menos 'margin'.
    """
    
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Calcula la perdida contrastiva de triplete.
        
        La distancia de coseno se define como D_cos(x, y) = 1.0 - CosineSimilarity(x, y).
        Como los embeddings de entrada ya estan normalizados L2, la similitud es simplemente el producto escalar.
        """
        # Distancia euclidea o del coseno simplificada al producto escalar si estan normalizados.
        # F.cosine_similarity es mas robusto ante fluctuaciones de precision.
        sim_pos = F.cosine_similarity(anchor, positive, dim=1)
        sim_neg = F.cosine_similarity(anchor, negative, dim=1)
        
        dist_pos = 1.0 - sim_pos
        dist_neg = 1.0 - sim_neg
        
        # Formula de Triplete: Loss = max(D(a,p) - D(a,n) + margin, 0)
        losses = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        
        return losses.mean()
