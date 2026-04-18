import torch
import torch.nn.functional as F
from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd

torch.set_num_threads(1)

MODEL_DIR = "triage_bert_model"
TOKENIZER_DIR = "triage_bert_tokenizer"
DATASET_PATH = "dataset.csv"
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

df = pd.read_csv(DATASET_PATH)
le = LabelEncoder()
le.fit(df["triage_level"].astype(str))

def triage_explanation(label):
    return {
        "High": "Critical condition detected. Immediate medical attention required.",
        "Medium": "Moderate urgency. Timely medical review recommended.",
        "Low": "Low urgency. Routine monitoring is sufficient."
    }.get(label, "")

def confidence_warning(conf):
    if conf < 0.5:
        return "Low confidence – manual review required."
    elif conf < 0.7:
        return "Medium confidence – clinician verification advised."
    return "High confidence prediction."

def keyword_highlights(text):
    words = [w.strip(",.!?") for w in text.lower().split()]
    unique = []
    for w in words:
        if w not in unique and len(w) > 2:
            unique.append(w)
    return unique[:10]

def predict_triage(text):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**encoding).logits
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_id = int(probs.argmax())
    label = le.inverse_transform([pred_id])[0]

    probabilities = {
        cls: round(float(probs[i]), 3)
        for i, cls in enumerate(le.classes_)
    }
    confidence = round(float(probs[pred_id]), 3)

    return label, probabilities, confidence

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = probabilities = explanation = warning = confidence = tokens = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("triage_text", "").strip()
        if user_text:
            prediction, probabilities, confidence = predict_triage(user_text)
            explanation = triage_explanation(prediction)
            warning = confidence_warning(confidence)
            tokens = keyword_highlights(user_text)

    return render_template(
        "index.html",
        prediction=prediction,
        probabilities=probabilities,
        explanation=explanation,
        warning=warning,
        confidence=confidence,
        tokens=tokens,
        user_text=user_text
    )

if __name__ == "__main__":
    app.run(debug=True)
