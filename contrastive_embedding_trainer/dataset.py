from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    """
    Dataset especializado para el entrenamiento contrastivo con tripletes de texto.
    
    Toma una lista de tuplas conteniendo (Anchor, Positive, Negative) y tokeniza
    los textos en tiempo de ejecucion de forma perezosa (lazy evaluation) para evitar
    el desperdicio de memoria RAM en datasets extensos.
    """
    
    def __init__(self, triplets: List[Tuple[str, str, str]], tokenizer, max_length: int = 128) -> None:
        """
        Args:
            triplets: Lista de tuplas conteniendo (Anchor, Positive, Negative).
            tokenizer: El tokenizador pre-entrenado correspondiente al modelo (ej. DistilBertTokenizer).
            max_length: Longitud maxima de secuencia de tokens para truncar o rellenar.
        """
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.triplets)

    def _tokenize_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokeniza un string individual aplicando relleno (padding) y truncamiento.
        Devuelve los tensores input_ids y attention_mask limpios sin dimensiones extra.
        """
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # squeeze(0) elimina la dimension de lote (batch) dummy creada por return_tensors='pt'
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor_text, pos_text, neg_text = self.triplets[idx]
        
        anchor_ids, anchor_mask = self._tokenize_text(anchor_text)
        pos_ids, pos_mask = self._tokenize_text(pos_text)
        neg_ids, neg_mask = self._tokenize_text(neg_text)
        
        return {
            "anchor_ids": anchor_ids,
            "anchor_mask": anchor_mask,
            "pos_ids": pos_ids,
            "pos_mask": pos_mask,
            "neg_ids": neg_ids,
            "neg_mask": neg_mask
        }
