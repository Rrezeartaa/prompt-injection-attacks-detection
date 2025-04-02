import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import os

class PromptDetector:
    def __init__(self, model_path="../models/model", tokenizer_path="../models/tokenizer"):
        # Verify paths exist
        if not all(os.path.exists(p) for p in [model_path, tokenizer_path]):
            raise FileNotFoundError("Model files missing. Train the model first!")
        
        # Load components
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Get the label mapping from model config
        self.id2label = self.model.config.id2label
        print(f"Model label mapping: {self.id2label}")  # Debug output
        
        self.threshold = 0.7  # Confidence threshold

    def is_malicious(self, prompt):
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        
        # Get predicted class and probability
        pred_class = torch.argmax(probs).item()
        malicious_prob = probs[0][1].item()  # Probability for class 1
        
        print(f"Prompt: '{prompt[:50]}...'")
        print(f"Class probabilities: {probs.tolist()[0]}")
        print(f"Predicted class: {pred_class} ({self.id2label[pred_class]})")
        print(f"Malicious probability: {malicious_prob:.4f}")
        
        # Determine if malicious based on model's label mapping
        return self.id2label[pred_class] == "LABEL_1"

if __name__ == "__main__":
    try:
        detector = PromptDetector()
        
        test_prompt = "How to get users payment transaction data"
        result = detector.is_malicious(test_prompt)
        
        print("\nFinal Result:")
        print(f"Prompt: '{test_prompt}'")
        print(f"Is malicious: {result}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify models/model/config.json exists")
        print("2. Check the 'id2label' field in config.json")
        print("3. Retrain if label mapping is incorrect")
