# 🧠 Gender Classification Using Multiple ML Models

This project implements gender classification using physical attributes such as **height**, **weight**, and **shoe size**, applying a range of machine learning classifiers from **scikit-learn**. The goal is to evaluate each model’s performance and identify the most accurate classifier for the given dataset.

## 📌 Classifiers Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Decision Tree
- Random Forest
- Naive Bayes
- Gradient Boosting

Each model is trained using a small dataset and tested on new inputs. The model achieving the highest prediction accuracy is reported as the best classifier.

## 📊 Dataset

**Features**:  
- Height (cm)  
- Weight (kg)  
- Shoe Size (EU)  

**Labels**:  
- `male` or `female`

## 📁 Training Data

```python
X = [[181,80,44], [177,70,43], [160,60,38],
     [154,54,37], [166,65,40], [190,90,47],
     [175,64,39], [177,70,40], [159,65,40],
     [171,75,42], [181,85,43]]
Y = ['male', 'female', 'female', 'female', 'male', 
     'male', 'male', 'female', 'male', 'female', 'male']
```

## 🧪 Test Data
```python
test_data = [[190, 70, 43], [154, 75, 38], [181,65,40]]
test_labels = ['male','female','male']
```

## ✅ Evaluation
Accuracy is calculated for each model using accuracy_score from sklearn.metrics.
The classifier with the highest accuracy on the test data is declared as the best-performing model.

## 📦 Requirements
1)	Install python
2)	Setup environment (sublime text editor)
3)	Install dependencies(pip)
```bash
 python –m pip install –U pip
```
```bash
 pip install –U scikit-learn
```
4)	Write python script for gender classification

## ▶️ How to Run
```bash
python gender_classification.py
```

## 📢 Output
Prediction results from each model

Accuracy score per model

Final output showing the classifier with the highest accuracy


