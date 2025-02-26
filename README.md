# 📌 Sentence Contradiction Classification

## **Introduction**  
The goal of this project is to classify pairs of sentences as **"contradiction," "entailment," or "neutral"** based on their meaning. The task requires building a model that can understand semantic relationships between text pairs.

---

## **📂 1. Reading the Dataset**  
The dataset contains **multiple sentence pairs** with labels indicating their relationship.  

✅ **Filtered the dataset to include only English sentences.**  
✅ **Removed missing and duplicate values.**  

---

## **📊 2. Exploratory Data Analysis (EDA)**  
Before training the model, we analyzed the data:  
✅ **Checked label distribution to see how many contradiction, entailment, and neutral cases exist.**  
✅ **Visualized word distributions using graphs.**  
✅ **Generated statistical summaries of numerical features.**  

---

## **🔍 3. Text Preprocessing Pipeline**  
✅ **Removing HTML tags**  
✅ **Removing Punctuations**  
✅ **Performing Lemmatization**  
✅ **Performing Lowercase**  
✅ **Removing Stopwords**  
✅ **Decontracting words**  
✅ **Replacing special characters with their string symbols**  

---

## **⚙️ 4. Feature Engineering**  
✅ **Added key sentence-level features:**  
  - `sentence1_len`, `sentence2_len`, `common_word_count`, `word_overlap_ratio`  
✅ **Applied TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.**  
✅ **Extracted word embeddings to enhance semantic understanding.**  

---

## **🤖 5. Model Building & Training**  

The dataset was split into:  
✅ **80% Training Data**  
✅ **20% Test Data**  

We tested multiple machine learning models:  
✅ **Logistic Regression**  
✅ **Random Forest**  
✅ **XGBoost (XGB)**  
✅ **Support Vector Machine (SVM)**  
✅ **Artificial Neural Network (ANN) using Keras**  
✅ **LSTM for sequence-based learning**  
✅ **BERT for advanced NLP understanding**  

We also performed **Grid Search** for **hyperparameter tuning**.  

---

## **📈 6. Model Evaluation**  

| **Model**              | **Accuracy** | **F1-Score** |
|------------------------|-------------|--------------|
| Logistic Regression   | **45.05%**   | **44.92%**   |
| Random Forest        | **44.25%**   | **43.55%**   |
| XGB                  | **45.12%**   | **44.82%**   |
| ANN                  | **39.37%**   | **31.55%**   |
| Decision Tree        | **39.96%**   | **39.96%**   |

✅ **XGB performed the best**, achieving **45.12% accuracy**.  
✅ A **confusion matrix** was used to analyze model errors.  

---

## **🚀 7. Deployment**  
We deployed the model using **Streamlit**, allowing users to:  
✅ Enter **two sentences**  
✅ Get a **classification result (Contradiction, Entailment, or Neutral)**  

🔗 **Try the Web App:** [👉 Click Here](https://sentenceclassifier.streamlit.app/)  

![Application Screenshot](image.png)
