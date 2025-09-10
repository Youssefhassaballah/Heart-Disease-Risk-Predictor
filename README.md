readme_content = """# ❤️ Heart Disease Prediction Project

## 📌 Project Overview
This project implements a **Heart Disease Prediction System** using **Machine Learning** techniques.  
The dataset used is the **Cleveland Heart Disease dataset (UCI Repository)**.  

The workflow covers **data preprocessing, feature selection, supervised & unsupervised learning, hyperparameter tuning, and deployment with Streamlit**.

---

## 📂 Project Structure
Heart_Disease_Project/ 
│── data/ 
│   ├── heart_disease.csv 
│── notebooks/ 
│   ├── 01_data_preprocessing.ipynb 
│   ├── 02_pca_analysis.ipynb 
│   ├── 03_feature_selection.ipynb 
│   ├── 04_supervised_learning.ipynb 
│   ├── 05_unsupervised_learning.ipynb 
│   ├── 06_hyperparameter_tuning.ipynb 
│── models/ 
│   ├── final_model.pkl 
│── ui/ 
│   ├── app.py (Streamlit UI) 
│── deployment/ 
│   ├── ngrok_setup.txt 
│── results/ 
│   ├── evaluation_metrics.txt 
│── README.md 
│── requirements.txt 
│── .gitignore 

---

## ⚙ Steps & Workflow

### **Step 1 – Data Preprocessing**
- Load Cleveland dataset.  
- Handle missing values (`?` → NaN → imputation).  
- Encode categorical variables.  
- Save cleaned dataset as `cleaned_heart.csv`.

---

### **Step 2 – Dimensionality Reduction (PCA)**
- Scale features using `StandardScaler`.  
- Apply PCA to reduce dimensionality.  
- Plot:
  - Cumulative Variance Plot.  
  - Scatter Plot (PC1 vs PC2).  
- Selected **11 PCs** explaining ~95% of variance.

---

### **Step 3 – Feature Selection**
Techniques used:  
1. **Random Forest Feature Importance**  
2. **Recursive Feature Elimination (RFE)**  
3. **Chi-Square Test**

- Final selected features:
thalach, slope, ca, oldpeak, thal, cp, exang, sex
- Saved reduced dataset as `03_data_selected_features.csv`.

---

### **Step 4 – Supervised Learning**
Models trained:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-score  
- ROC Curve & AUC  

---

### **Step 5 – Unsupervised Learning (Clustering)**
1. **K-Means Clustering**  
 - Elbow Method → optimal K = 2  
 - Cluster scatter plots  
2. **Hierarchical Clustering**  
 - Dendrogram  
 - Agglomerative clustering  

✔ Compared cluster assignments with actual disease labels.

---

### **Step 6 – Hyperparameter Tuning**
- **Logistic Regression** → GridSearchCV  
- **Random Forest** → RandomizedSearchCV  

Best performance on Test Set:  
- **Random Forest (tuned):**  
- Accuracy = 90%  
- Recall = 92%  
- ROC-AUC = 0.95 ✅ (Best Model)

---

### **Step 7 – Model Export**
- Saved best performing model (`RandomForestClassifier`) as:  
models/final_model.pkl

---

### **Step 8 – Streamlit App**
- Developed `app.py` with **Streamlit UI**.  
- Allows user input for:
thalach, slope, ca, oldpeak, thal, cp, exang, sex
- Outputs:
- Predicted Class (High Risk / Low Risk)  
- Probability Score  

Run locally:
```bash
streamlit run app/app.py
