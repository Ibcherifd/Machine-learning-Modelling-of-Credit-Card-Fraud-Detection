# Machine-learning-Modelling-of-Credit-Card-Fraud-Detection
This project implements and evaluates machine learning models to detect fraudulent credit card transactions. The system is designed using anonymized transaction data, with a focus on achieving high recall while minimizing false positives. The final solution is deployed via a Streamlit web app for real-time prediction.

## 📁 Project Structure

- `app.py` – Streamlit app for fraud prediction
- `xgboost_model.pkl` – Trained XGBoost model
- `train_and_save_model.py` – Script to preprocess data, apply SMOTE, and train model
- `requirements.txt` – Dependencies list for setting up the environment
- `CreditCardFraudDetection.ipynb` – Jupyter notebook for model training and EDA
- `README.md` – This documentation

## 🚀 How to Run the App Locally

1. **Create a virtual environment** (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate  # on Windows
```

2. **Install required packages**:

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**:

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## 🧠 Machine Learning Models

The following models were implemented and evaluated:

- **Logistic Regression** – Baseline model
- **Random Forest** – Ensemble method to reduce overfitting
- **XGBoost** – Final model with highest F1 score and recall

Performance metrics include accuracy, precision, recall, F1-score, and confusion matrix.

## 📊 Evaluation Results (XGBoost)

- **Accuracy**: 99.97%
- **Precision**: 99.82%
- **Recall**: 99.98%
- **F1 Score**: 99.90%

## 🛠 Tools & Libraries

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit
- SMOTE (from imbalanced-learn)
- Matplotlib & Seaborn (for visualization)

## 🔒 Data Source

Anonymized dataset from a well-known credit card fraud detection benchmark dataset (Dal Pozzolo et al., 2015), containing transactions made by European cardholders in 2013.

## 👤 Author

**Ibrahima Cherif Diallo**  
*Final Year BSc Computer Science Project*

## 📌 License

This project is for academic purposes only.

