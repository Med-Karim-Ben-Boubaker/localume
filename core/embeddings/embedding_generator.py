from transformers import AutoTokenizer, AutoModel
import torch 
from typing import List, Dict, Any

class EmbeddingModel:
    def __init__(self, model_name : str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_list = []

    def embed_text(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    
    def add_embedding(self, embedding: List[float], metadata: Dict[str, Any]):
        print(f"Adding embedding: {metadata}:{embedding}")
        self.embedding_list.append({
            "embedding": embedding,
            "metadata": metadata
        })
    
    
