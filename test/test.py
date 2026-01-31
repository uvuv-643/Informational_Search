import torch
import random
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn

# Diagnostic logging for device availability
print("=" * 50)
print("DIAGNOSTIC INFORMATION:")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
print(f"PyTorch version: {torch.__version__}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Selected device: {device}")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

class SPLADE(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.bert_mlm = AutoModelForMaskedLM.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert_mlm(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        relu_logits = F.relu(logits)
        relu_logits = relu_logits * attention_mask.unsqueeze(-1)
        pooled, _ = torch.max(relu_logits, dim=1)
        pooled = torch.clamp(pooled, max=10.0)
        return torch.log1p(pooled + 1e-8)


model = SPLADE().to(device)

# FIX: Add map_location parameter to handle GPU->CPU loading
print("\nLoading model checkpoint...")
try:
    # This will map CUDA tensors to CPU if CUDA is not available
    checkpoint = torch.load('model.pt', map_location=device)
    model.load_state_dict(checkpoint)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Additional diagnostic: try to inspect the checkpoint
    print("\nAttempting to load with CPU mapping for inspection...")
    try:
        checkpoint = torch.load('model.pt', map_location='cpu')
        print("Checkpoint loaded to CPU successfully")
        # Check if it's a state dict or full model
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())[:5]}...")  # Show first 5 keys
    except Exception as e2:
        print(f"Failed to load even with CPU mapping: {e2}")
        raise

model.eval()

test_queries = [
    "what is python programming",
    "how to lose weight",
    "best restaurants in paris",
    "covid vaccine side effects",
    "machine learning tutorial",
    "climate change causes",
    "how to cook pasta",
    "bitcoin price prediction",
    "yoga benefits health",
    "electric cars pros cons"
]

reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

with torch.no_grad():
    for query in test_queries:
        tokens = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
        q_rep = model(tokens['input_ids'].to(device), tokens['attention_mask'].to(device))

        top_indices = torch.topk(q_rep[0], k=20).indices
        top_weights = torch.topk(q_rep[0], k=20).values
        
        print(f"\nQuery: {query}")
        print("Top tokens:", [(tokenizer.decode([idx]), f"{weight:.2f}") for idx, weight in zip(top_indices, top_weights)])
