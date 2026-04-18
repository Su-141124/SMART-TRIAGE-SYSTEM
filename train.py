import pandas as pd
import ast
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv("dataset.csv")

def join_tokens(token_list_str):
    try:
        tokens = ast.literal_eval(token_list_str)
        if isinstance(tokens, list):
            return " ".join(tokens)
        return ""
    except:
        return ""

df['text'] = df['processed_text'].apply(join_tokens)

le = LabelEncoder()
df['triage_label'] = le.fit_transform(df['triage_level'])

X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['triage_label'], 
    test_size=0.2, random_state=42, stratify=df['triage_label']
)

class TriageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

train_dataset = TriageDataset(X_train, y_train, tokenizer)
val_dataset = TriageDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = BertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=4
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

epochs = 3
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

model.save_pretrained("triage_bert_model")
tokenizer.save_pretrained("triage_bert_tokenizer")
print("BERT training complete. Model saved.")
