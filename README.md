🏥 Smart Triage System using NLP & Machine Learning:-
This project focuses on solving a real-world healthcare problem by automatically prioritizing patients based on their symptoms using Natural Language Processing (NLP).
The system uses a BERT-based model to understand user-input symptom descriptions and classify them into appropriate triage levels, helping improve decision-making speed and consistency.

🚀 What This Project Does:-
Accepts patient symptoms as text input
Processes the input using NLP techniques
Uses a pre-trained BERT model for contextual understanding
Predicts the triage priority level
Displays results through a simple web interface

🛠️ Tech Stack:-
Python
Flask
PyTorch
Hugging Face Transformers (BERT)
Pandas
Scikit-learn

📂 Project Structure:-
smart-triage-nlp-ml/

├── app.py
├── train.py
├── predict.py
├── dataset.csv
├── requirements.txt

├── templates/
│ └── index.html

├── triage_bert_model/
├── triage_bert_tokenizer/

⚙️ How to Run the Project:-
Clone the repository
git clone <your-repo-link>
cd smart-triage-nlp-ml
Create virtual environment
python -m venv venv
venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Run the application
python app.py
Open in browser
http://127.0.0.1:5000/

🧠 How It Works:-
The system uses BERT (Bidirectional Encoder Representations from Transformers), which understands the context of words instead of just keywords.
Example:
"chest pain and sweating" → high priority
"mild headache" → low priority
This improves prediction accuracy compared to traditional machine learning models.

📊 Features
Real-time symptom classification
Context-aware predictions using BERT
Simple and user-friendly interface
End-to-end ML pipeline (training to deployment)
🔍 Future Improvements
Use larger and more diverse medical datasets
Improve UI/UX design
Deploy the application on cloud platforms
Add multilingual support
