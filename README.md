# MNIST Classification Project

This project involves implementing several machine learning algorithms to classify digits from the MNIST dataset. The dataset used is in the npz format. The goal is to predict and classify handwritten digits based on their images.

#### Project Files
mnist(KNN).ipynb
This notebook implements the k-Nearest Neighbors (KNN) algorithm for classifying the MNIST dataset. KNN is a simple, instance-based learning algorithm that assigns a class to a data point based on the majority class among its k-nearest neighbors.

mnist(NN with keras)(MAIN).ipynb
This notebook uses a Neural Network (NN) built with the Keras library to classify the MNIST digits. Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow. This implementation includes the creation, training, and evaluation of a neural network model.

mnist(SVM).ipynb
This notebook implements the Support Vector Machine (SVM) algorithm for digit classification. SVM is a powerful supervised learning model that can perform linear and non-linear classification using a technique called the kernel trick.

mnist(desicionTree).ipynb
This notebook uses a Decision Tree classifier for the MNIST dataset. Decision Trees are a type of model used for both classification and regression tasks. They work by splitting the data into subsets based on the value of input features, leading to a tree-like model of decisions.

mnist(randomforest).ipynb
This notebook implements the Random Forest algorithm. Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the mode of the classes (classification) or mean prediction (regression) of the individual trees.

mnist.npz
This file contains the MNIST dataset in npz format, which is a zipped archive containing numpy arrays. The dataset includes 60,000 training images and 10,000 testing images of handwritten digits.
#### Project Description
In this project, I have implemented various artificial intelligence algorithms on the MNIST dataset to tackle prediction and classification challenges. The objective is to accurately guess the numbers represented by handwritten digit images.

How to Run the Notebooks
Clone the repository:


git clone https://github.com/iamalirezaebi/mnist-classification.git
cd mnist-classification
Install the required Python libraries:

You can use pip to install the necessary libraries. It is recommended to use a virtual environment.


pip install numpy pandas scikit-learn keras tensorflow matplotlib
Open the Jupyter Notebooks:

You can use Jupyter Notebook or JupyterLab to open the .ipynb files.


#### jupyter notebook
Run the Notebooks:

Open each notebook and run the cells to execute the code. Make sure that mnist.npz is in the same directory as the notebooks, 
as they load the dataset from this file.

#### Algorithms Implemented
k-Nearest Neighbors (KNN)
Neural Network (NN) with Keras
Support Vector Machine (SVM)
Decision Tree
Random Forest
Conclusion
This project showcases different machine learning approaches to solving the problem of digit classification using the MNIST 
dataset. Each algorithm has its strengths and provides a unique perspective on how to approach the problem of image classification.
