Image Colourisation using Machine Learning Models
This project explores the application of deep learning techniques to automatically colorize black-and-white (grayscale) images using Python and powerful ML libraries. The model is trained and tested using Google Colab for quick prototyping and ease of GPU access.


📖 About the Project

Image colourisation is the process of adding plausible colors to grayscale images. This project implements a deep learning-based approach for automating the colorisation of black and white images using Convolutional Neural Networks (CNNs).

The goal is to understand how machine learning models can learn context and generate colored images that are visually appealing and close to real-world representations.

⚙️ Tech Stack

Python

Google Colab

TensorFlow / Keras

NumPy

OpenCV

Matplotlib

scikit-learn

📁 Files

We need to upload a black and white image and then it will generate a colurised image



🧠 Model Architecture

Convolutional Neural Networks (CNNs)

Encoder-Decoder architecture

Activation: ReLU

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

🚀 How to Run

Open the Colab Notebook

Upload your dataset or use sample images

Run all cells to:

Preprocess the data

Train the CNN model

Predict and visualize colorised images

📊 Results

The model effectively adds colors to black-and-white images.

Colorization is not always accurate, but results are visually convincing.

Performance improves with larger and more diverse datasets.

🌍 Applications

Restoring old historical photographs

Black-and-white movie colorisation

Enhancing grayscale medical images

Artistic photo editing tools

🔮 Future Work

Use pre-trained networks (e.g., VGG, U-Net)

Train on larger datasets for improved generalization

Integrate GANs for more realistic results

Deploy as a web app with a simple upload interface
