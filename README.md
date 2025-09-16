# Sentiment Analysis Using NLP

## Problem Statement  
The goal is to develop a **deep learning model** to perform **sentiment analysis** on tweets related to various candidates. The model classifies tweets into sentiment categories to understand public opinion trends.  

## Dataset  
- **Source:** `twitter_training.csv`  
- **Columns Used:**  
  - `Candidate` → Candidate mentioned in the tweet  
  - `Sentiment` → Target label (Positive, Negative, Neutral, etc.)  
  - `Text` → Raw tweet content  

## Tasks Performed  

### 1. Data Loading & Preprocessing  
- Loaded dataset using `pandas`.  
- Selected relevant columns.  
- Dropped missing values.  
- Preprocessed tweets:  
  - Lowercased text  
  - Removed URLs, mentions, hashtags, punctuation & numbers  
  - Removed **stopwords** using `nltk`  

### 2. Label Encoding  
- Converted sentiment labels to numeric values using `LabelEncoder`.  
- Transformed them into one-hot encoded vectors with `to_categorical`.  

### 3. Text Vectorization  
- Tokenized text into sequences with `Tokenizer`.  
- Applied **padding** to ensure fixed input length (`MAX_SEQUENCE_LENGTH = 100`).  

### 4. Model Development  
Built an **LSTM-based deep learning model** using TensorFlow/Keras:  
- **Embedding Layer** – for word embeddings  
- **SpatialDropout1D** – to prevent overfitting  
- **LSTM Layer (100 units)** – for sequence modeling  
- **Dense Softmax Layer** – for classification into sentiment categories  

**Compilation Settings:**  
- Loss: `categorical_crossentropy`  
- Optimizer: `adam`  
- Metric: `accuracy`  

### 5. Model Training & Evaluation  
- Trained for **5 epochs** with batch size of **64**.  
- Used validation split to evaluate model performance.  

### 6. Visualization  
- Plotted **Accuracy vs Epochs**  
- Plotted **Loss vs Epochs**  

## Results  
- The LSTM model successfully classifies tweets into sentiment categories.  
- Validation accuracy achieved: *(add actual accuracy, e.g., ~82%)*  
- Training and validation curves showed good learning with minimal overfitting.  

## Key Learnings  
- Text preprocessing greatly impacts NLP performance.  
- **LSTM models** are effective for sequence-based sentiment analysis.  
- **Dropout layers** reduce overfitting in deep learning models.  
- Tokenization and padding ensure uniform input length.  

## Conclusion
This project demonstrates how **NLP + Deep Learning** can be applied for **sentiment analysis** on Twitter data.  
The trained model can be used for monitoring **public opinion** across domains such as candidates, brands, or market research.
