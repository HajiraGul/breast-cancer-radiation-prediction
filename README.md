# Breast-Cancer-Radiation-Prediction

A Machine Learning–based classification system implemented in Python to predict whether a breast cancer patient will undergo radiation treatment. This project compares multiple supervised learning algorithms and evaluates their predictive performance using statistical metrics and visualization techniques.

---

## Project Overview

This project focuses on applying Machine Learning techniques in the healthcare domain. Using clinical and demographic features from the UCI Breast Cancer dataset, the system predicts whether a patient will receive radiation therapy.

The project includes full data preprocessing, model training, evaluation, and interpretation. It also demonstrates how predictive analytics can support medical decision-making by identifying patterns in patient data.

---

## 🚀 Core Functionality

- **Data Preprocessing Pipeline**
  - Load dataset from CSV file
  - Assign column names
  - Handle missing values
  - Remove duplicates
  - Normalize categorical inconsistencies
  - Save cleaned dataset

- **Machine Learning Models**
  - Naive Bayes Classifier
  - Logistic Regression
  - Decision Tree Classifier

- **Performance Evaluation**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix visualization

- **Comparative Analysis**
  - Model performance comparison
  - Interpretation of classification results
  - Identification of best-performing algorithm

- **Healthcare Decision Support**
  - Predict radiation treatment likelihood
  - Provide interpretable outputs for clinical use

---

## 📊 Dataset Information

- **Source:** UCI Machine Learning Repository  
- **Instances:** 286  
- **Attributes:** 9 clinical features + 1 target variable  

### Key Features

- Age  
- Menopause  
- Tumor Size  
- Inv-Nodes  
- Node-Caps  
- Degree of Malignancy  
- Breast  
- Breast-Quadrant  
- Class  

### Target Variable

- `irradiat` (Yes / No)

---

## 📈 Model Performance

| Model                | Accuracy |
|----------------------|----------|
| Naive Bayes          | 80.48%   |
| Logistic Regression  | 80.48%   |
| Decision Tree        | 71.95%   |

### 🔍 Result Interpretation

Naive Bayes and Logistic Regression achieved the highest accuracy and demonstrated better overall predictive performance compared to the Decision Tree model. These results indicate that probabilistic and linear models effectively captured patterns in the dataset.

---

## 🧠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Visual Studio Code  

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/breast-cancer-radiation-prediction.git
```

### 2️⃣ Navigate to Project Directory

```bash
cd breast-cancer-radiation-prediction
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Project (Using VS Code)

1. Open the project folder in **Visual Studio Code**.
2. Open the main Python files (e.g., `AllModels.py`) or notebook file.
3. Ensure the correct Python interpreter is selected.
4. Click **Run** or execute the script using:

```bash
python AllModels.py
```

The model will train and display evaluation results in the terminal.

---

## 📌 Future Enhancements

- Implement Ensemble Models (Random Forest, Gradient Boosting)
- Hyperparameter Optimization
- Deploy as a Web-based Application
- Integrate real-world hospital datasets
- Add Explainable AI techniques (SHAP, LIME)

---

## ⚠️ Limitations

- Small dataset size
- Limited demographic diversity
- No time-series or longitudinal patient data
- Model performance may vary on unseen real-world datasets
---

## 📜 License

This project is developed for academic purposes.
