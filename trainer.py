import os
import logging
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm

from model import SiameseTransformer, TripletCosineLoss

# Configuracion basica del registrador de eventos (logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class EmbeddingTrainer:
    """
    Gestor de entrenamiento (Trainer) para la optimizacion contrastiva de embeddings.
    
    Encapsula el bucle de entrenamiento, retropropagacion, calculo de loss,
    evaluacion del conjunto de validacion y persistencia de checkpoints.
    """
    
    def __init__(
        self,
        model: SiameseTransformer,
        tokenizer: PreTrainedTokenizerBase,
        margin: float = 0.3,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        device: Optional[str] = None
    ) -> None:
        """
        Args:
            model: Instancia de SiameseTransformer a entrenar.
            tokenizer: Tokenizador utilizado para serializar junto al modelo.
            margin: Margen de separacion para la funcion de perdida TripletCosineLoss.
            lr: Tasa de aprendizaje (learning rate) para AdamW.
            weight_decay: Penalizacion L2 para regularizacion y evitar sobreajuste.
            device: Dispositivo de computo ('cpu', 'cuda', 'mps'). Si es None, se autodetecta.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.criterion = TripletCosineLoss(margin=margin)
        
        # Seleccion inteligente del dispositivo de computo
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Dispositivo de entrenamiento seleccionado: {self.device}")
        self.model.to(self.device)
        
        # Configuramos el optimizador AdamW (estandar en Transformers por su manejo de weight decay)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Ejecuta una unica epoca de entrenamiento sobre el lote de datos proporcionado.
        
        Returns:
            Promedio de la perdida (loss) obtenida en la epoca.
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Entrenando", leave=False):
            # Limpiamos gradientes acumulados
            self.optimizer.zero_grad()
            
            # Transferimos todos los tensores del lote al dispositivo activo
            anchor_ids = batch["anchor_ids"].to(self.device)
            anchor_mask = batch["anchor_mask"].to(self.device)
            pos_ids = batch["pos_ids"].to(self.device)
            pos_mask = batch["pos_mask"].to(self.device)
            neg_ids = batch["neg_ids"].to(self.device)
            neg_mask = batch["neg_mask"].to(self.device)
            
            # Paso forward independiente para cada elemento del triplete
            # Comparten los pesos del transformer base gracias a la estructura de la red siamesa
            anchor_emb = self.model(anchor_ids, anchor_mask)
            pos_emb = self.model(pos_ids, pos_mask)
            neg_emb = self.model(neg_ids, neg_mask)
            
            # Calculamos la perdida contrastiva basada en distancias de coseno
            loss = self.criterion(anchor_emb, pos_emb, neg_emb)
            
            # Retropropagacion de gradientes y actualizacion de pesos
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
        
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evalua la funcion de perdida sobre un conjunto de datos de validacion.
        
        Returns:
            Promedio de la perdida en el conjunto de validacion.
        """
        self.model.eval()
        total_loss = 0.0
        
        # Desactivamos el calculo de gradientes para agilizar la inferencia y ahorrar memoria RAM/VRAM
        with torch.no_grad():
            for batch in dataloader:
                anchor_ids = batch["anchor_ids"].to(self.device)
                anchor_mask = batch["anchor_mask"].to(self.device)
                pos_ids = batch["pos_ids"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)
                neg_ids = batch["neg_ids"].to(self.device)
                neg_mask = batch["neg_mask"].to(self.device)
                
                anchor_emb = self.model(anchor_ids, anchor_mask)
                pos_emb = self.model(pos_ids, pos_mask)
                neg_emb = self.model(neg_ids, neg_mask)
                
                loss = self.criterion(anchor_emb, pos_emb, neg_emb)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
        
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 3,
        output_dir: str = "model_output"
    ) -> Dict[str, List[float]]:
        """
        Bucle principal de entrenamiento para multiples epocas.
        
        Args:
            train_dataloader: Cargador de datos del conjunto de entrenamiento.
            val_dataloader: Cargador de datos opcional para validacion.
            epochs: Numero de epocas a entrenar.
            output_dir: Carpeta de destino para guardar el modelo final y checkpoints.
            
        Returns:
            Diccionario conteniendo el historial de perdidas de train y validacion.
        """
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_dataloader)
            history["train_loss"].append(train_loss)
            
            val_log_str = ""
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                history["val_loss"].append(val_loss)
                val_log_str = f" | Val Loss: {val_loss:.4f}"
                
                # Si es el mejor modelo hasta el momento en validacion, guardamos checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_dir = os.path.join(output_dir, "best_model")
                    self.save_model(checkpoint_dir)
                    logger.info(f"Nuevo mejor modelo guardado en {checkpoint_dir} con Val Loss: {val_loss:.4f}")
            
            logger.info(f"Epoca {epoch}/{epochs} | Train Loss: {train_loss:.4f}{val_log_str}")
            
        # Al final, guardamos siempre el ultimo estado en la raiz de output_dir
        self.save_model(output_dir)
        logger.info(f"Modelo final entrenado guardado exitosamente en {output_dir}")
        return history
        
    def save_model(self, path: str) -> None:
        """
        Guarda tanto los pesos del transformer base en formato de Hugging Face como los pesos
        de la arquitectura completa de la red siamesa y su tokenizador.
        """
        os.makedirs(path, exist_ok=True)
        
        # 1. Guardamos el transformer base para que pueda ser cargado facilmente con AutoModel.from_pretrained
        # Esto es muy útil si el usuario solo quiere utilizar el backbone encoder ajustado.
        self.model.transformer.save_pretrained(path)
        
        # 2. Guardamos el estado completo de la clase SiameseTransformer (que incluye wrappers personalizados)
        torch.save(self.model.state_dict(), os.path.join(path, "siamese_state.pt"))
        
        # 3. Guardamos el tokenizador original para garantizar la compatibilidad al recargar el pipeline
        self.tokenizer.save_pretrained(path)
        
        logger.info(f"Archivos de modelo y tokenizador guardados en: {path}")
