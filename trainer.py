"""Compatibilidad: el código vive en el paquete `contrastive_embedding_trainer`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `trainer`
directamente (otros repos de ai-core-infra lo hacen)."""

from contrastive_embedding_trainer.trainer import *  # noqa: F401,F403
from contrastive_embedding_trainer.trainer import EmbeddingTrainer  # noqa: F401
