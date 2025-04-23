# Iris Flower Classification 🌸🌼 using Neural Network

This project demonstrates a simple **neural network classifier** built using **TensorFlow (Keras)** to classify Iris flowers into three species based on their petal and sepal dimensions.

---

## 📚 Dataset

We use the famous **Iris dataset** which contains 150 samples from 3 flower species:

- Setosa
- Versicolor
- Virginica

Each flower has 4 features:
- Sepal length
- Sepal width
- Petal length
- Petal width

---

## 🧠 Model Architecture

- Type: Sequential model
- Layers: 
  - Dense layer (3 neurons) with ReLU activation
- Optimizer: `adam`
- Loss Function: `sparse_categorical_crossentropy`
- Epochs: 115
- Batch size: 2

---

## 📊 Results

The model achieved:
- **Validation Accuracy**: Up to `83.33%`
- **Test Accuracy**: `66.67%`

> Note: Accuracy may vary slightly with each run due to random initialization and data splitting.

---



