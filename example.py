import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import SiameseTransformer
from dataset import TripletDataset
from trainer import EmbeddingTrainer


def get_embedding(model: SiameseTransformer, tokenizer, text: str, device) -> torch.Tensor:
    """
    Obtiene el vector de embedding normalizado L2 para una frase individual.
    """
    model.eval()
    with torch.no_grad():
        encoded = tokenizer(
            text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        # El paso forward ya nos devuelve el embedding normalizado L2
        embedding = model(input_ids, attention_mask)
        return embedding.squeeze(0)


def calculate_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """
    Calcula la similitud de coseno entre dos tensores normalizados.
    Dado que estan normalizados L2, es simplemente el producto escalar.
    """
    return torch.dot(emb1, emb2).item()


def print_similarity_matrix(model: SiameseTransformer, tokenizer, sentences, device, stage_name: str) -> None:
    """
    Muestra de forma estructurada las similitudes de coseno cruzadas entre frases de prueba.
    """
    print(f"\n--- Similitudes de Coseno ({stage_name}) ---")
    embeddings = [get_embedding(model, tokenizer, s, device) for s in sentences]
    
    # Imprimir matriz cruzada
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            if i < j:
                sim = calculate_cosine_similarity(embeddings[i], embeddings[j])
                print(f"'{s1}' VS '{s2}' => Similitud: {sim:.4f}")


def main() -> None:
    print("Iniciando demostracion del Contrastive Embedding Trainer...")
    
    # 1. Configuracion de dispositivo y carga de modelo base
    # Usamos distilbert-base-uncased para ejecucion rapida y local.
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando modelo base 'distilbert-base-uncased' en dispositivo '{device}'...")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = SiameseTransformer("distilbert-base-uncased")
    model.to(device)
    
    # Frases de prueba para medir el comportamiento antes y despues del ajuste
    test_sentences = [
        "Escribir codigo limpio en lenguaje Python",             # Programacion A
        "Desarrollo y optimizacion de software con Python",       # Programacion B (Similar a A)
        "Receta tradicional para preparar una paella marinera"   # Cocina (Diferente a A y B)
    ]
    
    # Evaluamos similitudes iniciales
    print_similarity_matrix(model, tokenizer, test_sentences, device, "ANTES del entrenamiento")
    
    # 2. Definicion del dataset ficticio de tripletes (Anchor, Positive, Negative)
    # Cubrimos tres nichos tematicos diferenciados: Programacion, Cocina y Astronomia.
    triplets = [
        # Programacion
        (
            "Como escribir un bucle for eficiente en Python",
            "Sintaxis basica para iterar en colecciones usando Python",
            "Preparacion casera de masa de pizza italiana"
        ),
        (
            "Optimizacion de consultas SELECT en bases de datos SQL",
            "Como crear indices agrupados en MySQL o PostgreSQL",
            "La expansion acelerada del universo segun la cosmologia moderna"
        ),
        # Cocina
        (
            "Receta paso a paso para hacer tortilla de patatas",
            "Ingredientes tradicionales para cocinar tortilla española con cebolla",
            "Escribir algoritmos de busqueda binaria en lenguaje C"
        ),
        (
            "Mejores tecnicas de amasado y fermentacion para pan de masa madre",
            "Como hornear panes artesanales crujientes en casa",
            "Comandos de Git para realizar un merge sin conflictos"
        ),
        # Astronomia
        (
            "Estudio del ciclo de vida de las estrellas gigantes rojas",
            "Evolucion estelar y colapso de supernovas en el espacio",
            "Como batir claras de huevo a punto de nieve"
        ),
        (
            "Observacion astronomica de las lunas de Jupiter con telescopio",
            "Visualizacion de satelites naturales en el sistema solar",
            "Definicion de variables locales y globales en programacion C++"
        )
    ]
    
    print("\nInicializando dataset y cargador de PyTorch...")
    dataset = TripletDataset(triplets, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 3. Entrenamiento con el EmbeddingTrainer personalizado
    # Ajustamos hiperparametros conservadores para evitar el olvido catastrofico en un dataset tan pequeño.
    trainer = EmbeddingTrainer(
        model=model,
        tokenizer=tokenizer,
        margin=0.3,
        lr=2e-5,
        device=device
    )
    
    epochs = 4
    print(f"\nIniciando entrenamiento contrastivo por {epochs} epocas...")
    trainer.train(dataloader, epochs=epochs, output_dir="model_output")
    
    # 4. Evaluacion final del impacto del ajuste contrastivo
    # Deberiamos notar que las frases similares (A y B) aumentan su similitud de coseno,
    # mientras que la frase disimil (Cocina) disminuye o mantiene su baja similitud respecto a ellas.
    print_similarity_matrix(model, tokenizer, test_sentences, device, "DESPUES del entrenamiento")
    print("\nDemostracion finalizada con exito. Los artefactos resultantes estan en 'model_output/'.")


if __name__ == "__main__":
    main()
