Deep Learning for Sentiment Analysis (LSTM, GRU, and Bidirectional LSTM)
This repository contains a comprehensive Jupyter notebook that explores sequence-based Deep Learning models for multi-class Sentiment Analysis. The project utilizes TensorFlow and Keras to preprocess textual data and train several Recurrent Neural Network (RNN) architectures—namely LSTM, GRU, and Bidirectional LSTM—to classify comments into different sentiment categories.

📌 Project Overview
Sentiment analysis is a crucial application of Natural Language Processing (NLP). This project focuses on building an end-to-end pipeline:

Data Acquisition: Downloads the latest version of the sentiment dataset via kagglehub.

Data Cleaning & Exploration: Handles missing values and visualizes the distribution of sentiments using Seaborn.

Text Preprocessing: Tokenizes, cleans, and vectorizes raw text using the Keras TextVectorization layer and incorporates Embedding layers with masking.

Model Architecture Comparison: Builds and evaluates multiple specialized architectures designed for sequence data:

Long Short-Term Memory (LSTM)

Gated Recurrent Unit (GRU)

Bidirectional LSTM (BiLSTM)

📊 Dataset
The model utilizes the Sentiment Analysis Dataset hosted on Kaggle.

Source: abdelmalekeladjelet/sentiment-analysis-dataset

File Used: sentiment_data.csv

Target Classes: Multi-class sentiment identifiers (0, 1, 2) found in the Sentiment column.

Feature Column: Raw user text located in the Comment column.

🛠️ Requirements & Installation
To run this notebook locally or in a cloud environment (like Google Colab), make sure you have Python 3.x installed along with the required libraries.

You can install the dependencies via pip:

Bash
pip install pandas numpy matplotlib seaborn tensorflow kagglehub
🚀 Getting Started
Clone the Repository:

Bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Run the Notebook:
Launch Jupyter Notebook or JupyterLab to execute the file:

Bash
jupyter notebook LSTM_\&_GRU_\&_BILSTM.ipynb
Kaggle API Setup:
The script automatically downloads the dataset through kagglehub. If you are running it locally, make sure you have configured your Kaggle API credentials (kaggle.json) if prompted.

🧠 Model Configuration Brief
Text Vectorization: Standardizes text and enforces a fixed token timeline via output_sequence_length=30.

Embedding Layer: Utilizes mask_zero=True to ignore padding tokens during recurrent learning steps.

Loss Function: sparse_categorical_crossentropy (since targets are integer labels).

Optimizer: adam.

📈 Results & Visualizations
The notebook includes plots detailing:

Class distribution counts for the dataset (Sentiment imbalance check).

Training vs. Validation Loss curves.

Training vs. Validation Accuracy curves per epoch across different architectures.
