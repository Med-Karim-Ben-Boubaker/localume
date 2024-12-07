from transformers import AutoTokenizer, AutoModel
import torch 
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer

class EmbeddingModel:
    def __init__(self, model_name : str = "sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text using extracted keywords.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        # Extract unique keywords and join them
        #keywords = list(dict.fromkeys(self.extract_keywords(text)))  # Remove duplicates
        #keywords_text = " ".join(keywords)
        
        # Optional: print cleaned keywords for debugging
        print("\n########################################")
        print(f"Keywords: {text}")
        print("########################################")

        # Prepare inputs
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use mean pooling for better representation
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return embeddings.cpu().squeeze().tolist()
    
    def extract_keywords(self, content: str, num_keywords: int = 50) -> List[str]:
        """
        Extract the most important keywords and phrases from the content.
        
        Args:
            content: Input text to analyze
            num_keywords: Maximum number of keywords to return
            
        Returns:
            List of the most significant keywords and phrases without duplicates
            
        Raises:
            ValueError: If content is empty or invalid
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        # Configure TF-IDF with modified parameters
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=100,
            ngram_range=(1, 2),  # Include both single words and bigrams
            token_pattern=r'(?u)\b[a-zA-Z]+(?:-[a-zA-Z]+)*[a-zA-Z]\b'
        )

        try:
            cleaned_content = " ".join(
                word for word in content.split() 
                if any(c.isalpha() for c in word)
                and not any(c.isdigit() for c in word)
            )
            
            tfidf_matrix = vectorizer.fit_transform([cleaned_content])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            # Create a dictionary to store the highest scoring version of each word
            word_scores: Dict[str, float] = {}
            
            for word, score in zip(feature_names, scores):
                # Split compound words
                parts = word.split()
                
                # For single words or compound words, store only the highest scoring version
                if len(parts) == 1:
                    if word not in word_scores or score > word_scores[word]:
                        word_scores[word] = score
                else:
                    # For bigrams, only keep if score is significantly higher than individual words
                    individual_scores = [word_scores.get(part, 0) for part in parts]
                    if score > max(individual_scores) * 1.5:  # Threshold for keeping bigrams
                        word_scores[word] = score
                    
            # Sort by score and take top num_keywords
            ranked = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in ranked[:num_keywords] if word_scores[word] > 0.1]

        except Exception as e:
            raise ValueError(f"Failed to extract keywords: {str(e)}")
    