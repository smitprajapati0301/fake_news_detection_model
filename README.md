📰 Fake News Detection using Deep Learning

This project implements a Deep Learning-based Fake News Detection system that classifies news articles as Real or Fake using Natural Language Processing (NLP) techniques.

🚀 Features
Detects whether a news article is Fake or Real
Uses Deep Learning (Embedding + Neural Network)
Trained on a large dataset (~44,000 articles)
Provides probability-based prediction
Interactive Streamlit dashboard
Clean and modular project structure
🧠 Project Workflow
Input Text
   ↓
Text Preprocessing
   ↓
Tokenization
   ↓
Padding
   ↓
Deep Learning Model
   ↓
Prediction (Fake / Real)
📂 Project Structure
fake-news-detection/
│
├── dataset/
│   ├── Fake.csv
│   ├── True.csv
│
├── model/
│   └── tokenizer.pkl
│
├── preprocessing/
│
├── train_model.py
├── evaluate_model.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
📊 Model Performance
Metric	Value
Accuracy	~99%
Precision	~0.99
Recall	~0.99
F1 Score	~0.99
⚙️ Technologies Used
Python
TensorFlow / Keras
Natural Language Processing (NLP)
Scikit-learn
Streamlit
📥 Dataset
ISOT Fake News Dataset
Contains real and fake news articles
~44,000 samples

📌 Note: Large dataset files may not be included due to GitHub size limits.

🛠️ Installation

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install dependencies:

pip install -r requirements.txt
▶️ How to Run
1️⃣ Train Model (optional)
python train_model.py
2️⃣ Evaluate Model
python evaluate_model.py
3️⃣ Run Prediction
python predict.py
4️⃣ Run Dashboard
python -m streamlit run app.py
🌐 Deployment

The application can be deployed using Streamlit Cloud to provide a web-based interface for real-time predictions.

⚠️ Limitations
May struggle with very short or ambiguous text
Dataset-specific patterns may affect generalization
Does not use social media metadata
🔮 Future Improvements
Use Transformer models (BERT, RoBERTa)
Add real-time news API integration
Improve UI/UX of dashboard
Support multilingual news
👨‍💻 Author

Smit Prajapati

📌 Conclusion

This project demonstrates how deep learning and NLP techniques can be used to detect fake news effectively. The system achieves high accuracy and provides a user-friendly interface for real-time predictions.

⭐ If you like this project

Give it a ⭐ on GitHub!
