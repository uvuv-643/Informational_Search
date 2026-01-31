import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import ir_datasets

class MSMARCODataset(Dataset):
    def __init__(self, tokenizer, max_length=128, max_items=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        dataset = ir_datasets.load("msmarco-passage/train/triples-small")
        
        queries = {}
        docs = {}
        for q in dataset.queries_iter():
            queries[q.query_id] = q.text
        for d in dataset.docs_iter():
            docs[d.doc_id] = d.text
        
        self.data = []
        count = 0
        for item in dataset.docpairs_iter():
            if max_items and count >= max_items:
                break
            self.data.append({
                'query': queries[item.query_id],
                'pos_doc': docs[item.doc_id_a],
                'neg_doc': docs[item.doc_id_b]
            })
            count += 1
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = self.tokenizer(item['query'], truncation=True, padding='max_length', 
                               max_length=self.max_length, return_tensors='pt')
        pos_doc = self.tokenizer(item['pos_doc'], truncation=True, padding='max_length',
                                 max_length=self.max_length, return_tensors='pt')
        neg_doc = self.tokenizer(item['neg_doc'], truncation=True, padding='max_length',
                                 max_length=self.max_length, return_tensors='pt')
        return {
            'q_ids': query['input_ids'].squeeze(),
            'q_mask': query['attention_mask'].squeeze(),
            'p_ids': pos_doc['input_ids'].squeeze(),
            'p_mask': pos_doc['attention_mask'].squeeze(),
            'n_ids': neg_doc['input_ids'].squeeze(),
            'n_mask': neg_doc['attention_mask'].squeeze()
        }

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

def compute_loss(q_reps, p_reps, n_reps, lambda_q=1e-3, lambda_d=1e-4, tau=0.1):
    batch_size = q_reps.size(0)
    all_docs = torch.cat([p_reps, n_reps], dim=0)
    
    q_reps = F.normalize(q_reps, p=2, dim=-1)
    all_docs = F.normalize(all_docs, p=2, dim=-1)
    
    scores = torch.matmul(q_reps, all_docs.T) / tau
    scores = torch.clamp(scores, min=-100, max=100)
    
    labels = torch.arange(batch_size, device=q_reps.device)
    ce_loss = F.cross_entropy(scores, labels)
    
    l1_q = lambda_q * torch.mean(torch.abs(q_reps))
    l1_d = lambda_d * torch.mean(torch.abs(all_docs))
    
    total_loss = ce_loss + l1_q + l1_d
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        return torch.tensor(0.0, device=q_reps.device, requires_grad=True)
    
    return total_loss

parser = argparse.ArgumentParser(prog='splade_training')
parser.add_argument('-r', '--result', required=True, help='Output file')

if __name__ == '__main__':
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    print('Loading dataset...')
    dataset = MSMARCODataset(tokenizer, max_items=3 * 32000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print('Training set has {} instances'.format(len(dataset)))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPLADE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    scaler = torch.cuda.amp.GradScaler()
    
    model.train()
    for epoch in range(1):
        print('EPOCH {}:'.format(epoch + 1))
        total_loss = 0.0
        running = 0.0
        valid_steps = 0
        
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            
            try:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    q_reps = model(batch['q_ids'].to(device), batch['q_mask'].to(device))
                    p_reps = model(batch['p_ids'].to(device), batch['p_mask'].to(device))
                    n_reps = model(batch['n_ids'].to(device), batch['n_mask'].to(device))
                    loss = compute_loss(q_reps, p_reps, n_reps)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    loss_val = loss.item()
                    total_loss += loss_val
                    running += loss_val
                    valid_steps += 1
                else:
                    scaler.update()
                    continue
                    
            except RuntimeError as e:
                scaler.update()
                continue
            
            if (i + 1) % 100 == 0 and valid_steps > 0:
                print('  batch {} loss: {}'.format(i + 1, running/valid_steps))
                running = 0.0
                valid_steps = 0
        
        if len(dataloader) > 0:
            print('Epoch {} avg loss = {}'.format(epoch + 1, total_loss / len(dataloader)))
    
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
    
    with torch.no_grad():
        for query in test_queries:
            tokens = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
            q_rep = model(tokens['input_ids'].to(device), tokens['attention_mask'].to(device))
            
            top_indices = torch.topk(q_rep[0], k=20).indices
            top_weights = torch.topk(q_rep[0], k=20).values
            
            print('\nQuery: {}'.format(query))
            print('Top tokens: {}'.format([(tokenizer.decode([idx]), '{:.2f}'.format(weight.item())) 
                                          for idx, weight in zip(top_indices, top_weights)]))
    
    model_path = 'splade_checkpoint.pt'
    torch.save(model.state_dict(), model_path)
    
    print('Move {} to {}'.format(model_path, args.result))
    shutil.move(model_path, args.result)
