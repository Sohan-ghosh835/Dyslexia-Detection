# Dyslexia Detection System

A Machine Learning-based system designed to assist in early screening and detection of dyslexia-related patterns through structured data analysis and predictive modeling.

---

## 1. Project Overview

The Dyslexia Detection System leverages machine learning algorithms to analyze handwriting or structured input features and predict potential indicators of dyslexia.

The project demonstrates:

- Data preprocessing and cleaning
- Feature engineering
- Model training and evaluation
- Model serialization and reuse
- Reproducible experimentation through notebooks

This repository contains the trained model, development notebooks, and supporting datasets required to replicate results.

---

## 2. Problem Statement

Dyslexia is a learning disorder that affects reading and writing abilities. Early detection can significantly improve intervention outcomes.

This project aims to build a computational model that can:

- Analyze input features related to handwriting or structured patterns
- Learn distinguishing characteristics
- Predict classification labels with measurable accuracy

---

## 3. Repository Structure


Dyslexia-Detection/
│
├── Dyslexia_Frontend.ipynb # Frontend interface notebook
├── dyslexia_detection.ipynb # Model development notebook
├── Dyslexia_classifier.h5 # Trained ML model
├── Dyslexia_classification.zip # Dataset
└── README.md


---

## 4. Tech Stack

- Python
- NumPy
- Pandas
- TensorFlow / Keras (for .h5 model)
- Scikit-learn (evaluation metrics)
- Jupyter Notebook

---

## 5. Installation & Setup

### Step 1: Clone the repository


git clone https://github.com/your-username/Dyslexia-Detection.git

cd Dyslexia-Detection


### Step 2: Create a virtual environment


python -m venv venv


Activate it:

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate


### Step 3: Install required dependencies


pip install numpy pandas tensorflow scikit-learn matplotlib jupyter


---

## 6. Model Loading Example

If using the trained `.h5` model:


from tensorflow.keras.models import load_model
import numpy as np

model = load_model("Dyslexia_classifier.h5")

Example input (replace with actual feature vector)

sample_input = np.array([[0.2, 0.5, 0.8, 0.1]])

prediction = model.predict(sample_input)

print("Prediction:", prediction)


---

## 7. Model Training Workflow (Simplified Example)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

Split dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

Define model

model = Sequential([
Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
Dense(32, activation='relu'),
Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])

Train model

model.fit(X_train, y_train, epochs=20, batch_size=32)

Evaluate

loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)


---

## 8. Evaluation Metrics

The system evaluates performance using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Example:


from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


---

## 9. How to Run the Project

Option 1: Run Jupyter Notebook


jupyter notebook


Open:
- `dyslexia_detection.ipynb` for model development
- `Dyslexia_Frontend.ipynb` for frontend interaction

Option 2: Use trained model directly in a Python script.

---

## 10. Future Improvements

- Larger and more diverse dataset
- Hyperparameter optimization
- Cross-validation
- Deployment as a web application
- Real-time handwriting input support

---

## 11. Contributors

- Sohan Ghosh  
- Soumyajit Chakraborty  

---

## 12. License

This project is intended for academic and research purposes.