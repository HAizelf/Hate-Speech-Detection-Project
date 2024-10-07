
# Hinglish-Hindi Hateful and Offensive Comments Detection

**Contributors**  
- Hitesh Bhandari (IIIT Delhi)  
- Ananya Dabas (IIIT Delhi)  
- Aaloke Mozumdar (IIIT Delhi)  
- Akshita Gupta (IIIT Delhi)

## Overview

This project focuses on detecting hateful and offensive comments in Hinglish (Hindi-English code-mixed) text, a pressing issue in social media discourse. We leveraged the HASOC 2021 Hinglish dataset for binary classification (Hateful/Offensive vs. Non-Hateful) and experimented with both traditional machine learning and transformer-based models. The aim is to improve the detection of hate speech and offensive language in the complex multilingual context of India.

## Problem Statement

The rise of social media in India has given a platform for public discourse, which unfortunately includes hate speech targeting minority communities. Existing hate detection systems face challenges due to the prevalence of code-mixed languages like Hinglish, making it difficult for traditional models to handle. Our system aims to detect and prevent the spread of hate speech by using advanced machine learning and deep learning models.

## Dataset

We used the **HASOC 2021 Hinglish dataset**, which is split as follows:

| Split | Samples Count |
|-------|---------------|
| Train | 6833          |
| Val   | 760           |
| Test  | 844           |

**Classes**:
- **HOF**: Hateful and Offensive comments
- **NOT**: Non-Hateful and Offensive comments

### Preprocessing Techniques:
- Lowercasing
- Removal of HTML tags, URLs, usernames, extra whitespaces
- Contraction and punctuation removal
- Lemmatization and stopword removal
- Tokenization

## Methodology

### 1. Machine Learning Model (XGBoost)
- **Feature Extraction**: TF-IDF for text representation
- **Classifier**: XGBoost

### 2. Transformer-based Models
- **BERT-based Models**: Fine-tuned on our dataset for hate speech detection. We experimented with various models:
    - **RoBERTa**
    - **DistilBERT**
    - **XLM-RoBERTa** (multilingual)
    - **HingRoBERTa** (trained on Hinglish code-mixed data)
    - **HingRoBERTa-mixed** (trained on both Roman and Devanagari scripts)

## Results and Analysis

| Model               | Learning Rate | F1-Score |
|---------------------|---------------|----------|
| TF-IDF + XGBoost    | 0.5           | 74.2%    |
| RoBERTa             | 5e-7          | 77.5%    |
| DistilBERT          | 5e-7          | 76.1%    |
| HingRoBERTa         | 2e-5          | 80.0%    |
| HingRoBERTa-mixed   | 2e-6          | 77.6%    |

- **Best Model**: HingRoBERTa achieved the highest F1 score of 80% on the test set.
- **Observations**: Multilingual models, especially HingRoBERTa, outperformed monolingual models due to better alignment with the code-mixed dataset.

## Conclusion

Our experiments demonstrate the efficacy of multilingual transformer models, particularly HingRoBERTa, in detecting hate speech in Hinglish text. These models outperform traditional machine learning models and monolingual BERT variants.

## Future Work

Future work could focus on:
- Expanding the dataset with more code-mixed samples.
- Exploring other transformer-based architectures like GPT for better contextual understanding.
- Real-time deployment for hate speech detection on social media platforms.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/hate-detection-hinglish
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Preprocess the dataset and train the models:
    ```bash
    python train.py
    ```
4. Evaluate the model on the test set:
    ```bash
    python evaluate.py
    ```

## References

For a list of references used in the project, please check the [report](./report.pdf).
