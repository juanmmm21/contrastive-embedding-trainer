"""Compatibilidad: el código vive en el paquete `contrastive_embedding_trainer`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `model`
directamente (otros repos de ai-core-infra lo hacen)."""

from contrastive_embedding_trainer.model import *  # noqa: F401,F403
from contrastive_embedding_trainer.model import SiameseTransformer, TripletCosineLoss  # noqa: F401
