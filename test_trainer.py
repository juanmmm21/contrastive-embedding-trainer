import unittest
import torch
import tempfile
import shutil
import os
from typing import Dict

from transformers import AutoTokenizer, DistilBertConfig, DistilBertModel

from model import SiameseTransformer, TripletCosineLoss
from dataset import TripletDataset
from trainer import EmbeddingTrainer


class TestContrastiveEmbeddingTrainer(unittest.TestCase):
    """
    Casos de prueba unitarios para verificar la integridad y funcionalidad
    de todos los componentes del contrastive-embedding-trainer.
    """
    
    @classmethod
    def setUpClass(cls) -> None:
        # Cargar primero el tokenizador para conocer el tamaño real del vocabulario
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        except Exception:
            cls.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=False)
            
        # Creamos una configuracion muy reducida y aleatoria del modelo para evitar descargas
        # desde internet y agilizar las pruebas en local de forma aislada.
        cls.config = DistilBertConfig(
            vocab_size=cls.tokenizer.vocab_size,
            dim=64,             # Reducimos dimension para rapidez de calculo
            n_layers=1,         # Una sola capa es suficiente para verificar el flujo
            n_heads=2,
            hidden_dim=128,
            dropout=0.0
        )

    def setUp(self) -> None:
        # Inicializamos la red siamesa con pesos aleatorios basados en la mini-configuracion
        self.raw_transformer = DistilBertModel(self.config)
        self.model = SiameseTransformer()
        # Sobreescribimos el transformer interno con nuestro transformer aleatorio diminuto
        self.model.transformer = self.raw_transformer
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_siamese_transformer_forward(self) -> None:
        """
        Verifica que el paso forward de SiameseTransformer produzca embeddings de longitud
        correcta y que esten normalizados L2 (su norma euclidea debe ser aproximadamente 1.0).
        """
        batch_size = 4
        seq_length = 32
        
        # Generamos entradas ficticias
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length))
        
        embeddings = self.model(input_ids, attention_mask)
        
        # Comprobar dimensiones de salida: (batch_size, embedding_dimension_del_modelo)
        self.assertEqual(embeddings.shape, (batch_size, 64))
        
        # Comprobar la normalizacion L2: la norma de cada vector debe ser 1.0
        norms = torch.norm(embeddings, p=2, dim=1)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_triplet_cosine_loss(self) -> None:
        """
        Verifica el comportamiento de la funcion de perdida TripletCosineLoss.
        """
        loss_fn = TripletCosineLoss(margin=0.3)
        
        # Caso 1: Perfecto. Anchor esta muy cerca de Positive y muy lejos de Negative.
        # Similitud Anchor-Pos = 1.0 (Distancia = 0)
        # Similitud Anchor-Neg = -1.0 (Distancia = 2)
        # Loss = clamp(0.0 - 2.0 + 0.3, min=0) = clamp(-1.7, min=0) = 0.0
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[1.0, 0.0]])
        negative = torch.tensor([[-1.0, 0.0]])
        
        loss_perfect = loss_fn(anchor, positive, negative)
        self.assertEqual(loss_perfect.item(), 0.0)
        
        # Caso 2: Malo. Anchor esta lejos de Positive y cerca de Negative.
        # Similitud Anchor-Pos = -1.0 (Distancia = 2)
        # Similitud Anchor-Neg = 1.0 (Distancia = 0)
        # Loss = clamp(2.0 - 0.0 + 0.3, min=0) = 2.3
        anchor_bad = torch.tensor([[1.0, 0.0]])
        positive_bad = torch.tensor([[-1.0, 0.0]])
        negative_bad = torch.tensor([[1.0, 0.0]])
        
        loss_bad = loss_fn(anchor_bad, positive_bad, negative_bad)
        self.assertAlmostEqual(loss_bad.item(), 2.3, places=5)

    def test_triplet_dataset(self) -> None:
        """
        Verifica la tokenizacion e indexacion correcta del TripletDataset.
        """
        triplets = [
            ("Astronomía y estrellas", "El cosmos y el espacio", "Cómo hacer una tortilla de patatas"),
            ("Aprender a programar", "Desarrollo de software en Python", "Receta de pollo al horno")
        ]
        
        dataset = TripletDataset(triplets, self.tokenizer, max_length=16)
        
        self.assertEqual(len(dataset), 2)
        
        item = dataset[0]
        # Verificar que contiene las llaves requeridas para los tripletes
        expected_keys = {"anchor_ids", "anchor_mask", "pos_ids", "pos_mask", "neg_ids", "neg_mask"}
        self.assertEqual(set(item.keys()), expected_keys)
        
        # Verificar dimensiones correspondientes a max_length
        for key in expected_keys:
            self.assertEqual(item[key].shape, (16,))
            self.assertEqual(item[key].dtype, torch.long)

    def test_trainer_integration(self) -> None:
        """
        Realiza una ejecucion de entrenamiento simulada (integration test)
        para asegurar que el optimizador actualiza los parametros.
        """
        triplets = [
            ("Python programming", "Coding in python", "Cooking pasta recipe"),
            ("Deep learning model", "Training neural networks", "Guitar chords for beginners"),
            ("How to bake bread", "Baking homemade sourdough", "SQL query optimization tips"),
            ("Quantum computing basics", "Qubits and quantum gates", "Gardening indoor plants")
        ]
        
        dataset = TripletDataset(triplets, self.tokenizer, max_length=16)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Instanciamos el trainer
        trainer = EmbeddingTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            margin=0.3,
            lr=1e-3, # Alta tasa para evidenciar cambios en parametros
            device="cpu"
        )
        
        # Capturamos una copia de los parametros antes de entrenar
        initial_params = [p.clone() for p in self.model.parameters() if p.requires_grad]
        
        # Entrenamos una epoca
        history = trainer.train(dataloader, epochs=1, output_dir=self.tmp_dir)
        
        # Verificar que se registro perdida de entrenamiento
        self.assertEqual(len(history["train_loss"]), 1)
        self.assertGreater(history["train_loss"][0], 0.0)
        
        # Verificar que los parametros sufrieron modificaciones en el paso optimizador
        updated_params = [p for p in self.model.parameters() if p.requires_grad]
        param_changed = False
        for p_init, p_upd in zip(initial_params, updated_params):
            if not torch.equal(p_init, p_upd):
                param_changed = True
                break
                
        self.assertTrue(param_changed, "Los parametros del modelo no se actualizaron durante el entrenamiento.")
        
        # Verificar que se crearon los archivos de guardado esperados
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "siamese_state.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "tokenizer_config.json")))


if __name__ == "__main__":
    unittest.main()
