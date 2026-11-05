"""Contrastive (triplet) fine-tuning of transformer sentence embeddings."""

from .model import SiameseTransformer
from .model import TripletCosineLoss
from .dataset import TripletDataset
from .trainer import EmbeddingTrainer

__version__ = "0.1.0"

__all__ = ['SiameseTransformer', 'TripletCosineLoss', 'TripletDataset', 'EmbeddingTrainer']
