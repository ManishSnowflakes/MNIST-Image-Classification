# **MNIST Image Classification**

**I have using A Deep Learning Approach to Handwritten Digit Recognition using the MNIST Dataset**

---

## **📌 Project Overview**
This project demonstrates a deep learning solution to the classic **MNIST image classification problem**, where the goal is to classify handwritten digits (0-9) from grayscale images. Using neural networks, we achieve accurate predictions, showcasing foundational concepts in image classification and model training.

---

## **🌟 Key Features**
- Utilizes the popular **MNIST dataset** as a benchmark for handwritten digit recognition.
- Implements a **neural network** architecture using modern deep learning frameworks.
- Comprehensive **data preprocessing** and augmentation techniques.
- Detailed **model training, validation, and evaluation** pipeline.
- Performance insights with visualizations like confusion matrices and loss/accuracy graphs.

---

## **🗂 Project Structure**
The repository is organized as follows:

```
MNIST-Image-Classification/
├── data/
│   ├── mnist_train.csv  # Training data
│   ├── mnist_test.csv   # Test data
├── notebook/
│   ├── 01_mnist.ipynb   # Main notebook containing the project code
├── readme/
│   ├── README.md        # This README file
```

---

## **📊 Dataset**
The **MNIST dataset** consists of:
- **60,000 training images**: Grayscale 28x28 pixel images of handwritten digits.
- **10,000 test images**: Similar structure for testing.
  
The dataset is widely used as a benchmark in machine learning and is available via **TensorFlow/Keras datasets** or other sources like **Kaggle**.

---

## **⚙️ Model Architecture**
This project employs a **neural network** for classification:
1. **Input Layer**: Flattened 28x28 grayscale image.
2. **Hidden Layers**: Fully connected dense layers with ReLU activation.
3. **Output Layer**: Softmax layer predicting one of 10 classes (0-9).

The model is trained using **categorical cross-entropy loss** and **Adam optimizer** for efficient learning.

---

## **📈 Results**
- **Training Accuracy**: Achieved ~98% on the training dataset.
- **Validation Accuracy**: Achieved ~97% on the test dataset.
- **Evaluation Metrics**: Precision, recall, and F1-score were calculated for detailed insights.

---

## **🛠️ Skills Gained**
Through this practical experience, I developed and enhanced the following skills:
- **Data Handling**: Loading, cleaning, and preprocessing datasets for machine learning models.
- **Deep Learning**: Designing and training neural networks with multiple layers for image classification tasks.
- **Model Evaluation**: Utilizing metrics like accuracy, precision, recall, and F1-score to evaluate model performance.
- **Data Visualization**: Creating plots to analyze data distribution, loss trends, and model accuracy.
- **Optimization Techniques**: Implementing optimization algorithms such as **Adam** and strategies to prevent overfitting like **dropout layers**.
- **Framework Proficiency**: Working with frameworks like **TensorFlow**, **Keras**, and **Matplotlib**.
- **Python Programming**: Enhancing coding skills with Python, focusing on libraries like NumPy, Pandas, and TensorFlow.
- **Project Organization**: Structuring files, notebooks, and documentation for better collaboration and readability.

---

## **🚀 How to Use**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ManishSnowflakes/MNIST-Image-Classification.git
   cd MNIST-Image-Classification
   ```
2. **Install Dependencies**:
   Ensure you have Python installed and run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   Open the `01_mnist.ipynb` notebook and run all cells:
   ```bash
   jupyter notebook notebook/01_mnist.ipynb
   ```
4. **View Results**:
   Check the visualizations, metrics, and model performance.

---

## **📚 Key Learnings**
1. **Data Preprocessing**: Importance of normalizing pixel values for faster convergence.
2. **Neural Networks**: Basics of dense layers, activation functions, and overfitting prevention using dropout.
3. **Evaluation**: Understanding metrics like accuracy, precision, and recall for classification tasks.

---

## **🔗 References**
- [MNIST Dataset on Kaggle](https://www.kaggle.com/c/digit-recognizer)
- [TensorFlow MNIST Dataset Guide](https://www.tensorflow.org/datasets/catalog/mnist)
- [Deep Learning Book by Ian Goodfellow](https://www.deeplearningbook.org/)

---

## **🤝 Contributing**
Contributions are welcome! If you find issues or want to enhance the project, feel free to fork the repository, make changes, and submit a pull request.

---

## **📧 Contact**
For any queries or collaborations, feel free to reach out:
- **Name**: Manish Khobragade
- **Email**: [manishsnowflakes@gmail.com](mailto:manishsnowflakes@gmail.com)
- **LinkedIn**: [Manish Khobragade](https://www.linkedin.com/in/manishkhobragade-itengineer)

---

Let me know if there are any additional details you'd like to include!
