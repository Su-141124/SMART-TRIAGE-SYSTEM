import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd

MODEL_DIR = "triage_bert_model"           
TOKENIZER_DIR = "triage_bert_tokenizer"    
DATASET_PATH = "dataset.csv"              
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset file not found at '{DATASET_PATH}'. This script rebuilds label mapping from the dataset. "
        "If you don't have the dataset, either save label_encoder during training or use the fallback script."
    )

df = pd.read_csv(DATASET_PATH)
if 'triage_level' not in df.columns:
    raise ValueError("Dataset does not contain 'triage_level' column required to rebuild label mapping.")

le = LabelEncoder()
le.fit(df['triage_level'].astype(str))  # ensure strings

def predict_triage(text: str) -> str:
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_id = int(torch.argmax(logits, dim=-1).cpu().numpy()[0])

    label = le.inverse_transform([pred_id])[0]
    return label

if __name__ == "__main__":
    text = input("Enter text to classify: ").strip()
    if not text:
        print("No input provided.")
    else:
        try:
            pred = predict_triage(text)
            print("\nPredicted TRIAGE LEVEL:", pred)
        except Exception as e:
            print("Prediction failed:", e)
