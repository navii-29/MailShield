# 🛡️ MailShield - FAKE MAIL/MESSAGE DETECTOR



MailShield is an end-to-end **machine learning web application** designed to detect phishing and spam emails using **Natural Language Processing (NLP)** and **supervised learning**.  
The system analyzes email content and predicts whether an email is **phishing (spam)** or **legitimate**, helping users stay safe from malicious attacks.

---

## 📌 Project Overview

Phishing emails remain one of the most common and dangerous cyber threats. MailShield aims to:

- Analyze email text content
- Learn linguistic patterns used in phishing
- Classify emails accurately using ML
- Provide a clean, user-friendly web interface for testing

The project covers the **full ML lifecycle**:
> data collection → preprocessing → vectorization → model training → evaluation → deployment with Flask.

---

## 📂 Dataset Overview (Research-Based)

The dataset used in this project is a **Phishing Email Dataset** compiled by researchers from multiple trusted sources to study phishing tactics and spam behavior.

### 🔹 Initial Datasets Used

#### 1. **Enron & Ling Datasets**
- Focus on **email subject lines and body text**
- Labeled as:
  - `spam / phishing`
  - `legitimate`
- Widely used in academic phishing detection research

#### 2. **CEAS Dataset**
- Contains real-world spam and legitimate emails
- Includes metadata such as sender and recipient

#### 3. **Nazario Phishing Corpus**
- Curated phishing emails collected from security research
- Focuses on social engineering techniques

#### 4. **Nigerian Fraud Dataset**
- Includes scam and fraud-related emails
- Useful for detecting financial and identity-based phishing

#### 5. **SpamAssassin Dataset**
- Popular benchmark dataset for spam filtering
- Includes both ham (legitimate) and spam emails

---

## 📊 Final Combined Dataset

All datasets were merged and cleaned to create a **single comprehensive corpus**.

| Category | Count |
|--------|-------|
| Total Emails | ~82,500 |
| Spam / Phishing | 42,891 |
| Legitimate | 39,595 |

This balanced dataset improves generalization and reduces model bias.

---

## 🧹 Data Preprocessing (NLP Pipeline)

To ensure high-quality input for the model, the following NLP preprocessing steps were applied:

- Lowercasing text
- Removal of:
  - Punctuation
  - Special characters
  - Numbers
- Tokenization
- Stopword removal
- **Lemmatization** (to reduce words to their base form)
- Whitespace normalization

These steps help reduce noise and improve semantic understanding.

---

## 🧠 Feature Engineering (Vectorization)

### 🔹 Word2Vec (Trained from Scratch)

Instead of using TF-IDF, **Word2Vec embeddings** were trained from scratch on the email corpus to capture:

- Semantic similarity
- Contextual word relationships
- Phishing-specific language patterns

Benefits:
- Dense vector representation
- Better semantic meaning compared to sparse vectors
- Domain-specific embeddings (email language)

---

## 🤖 Model Architecture

### 🔹 Random Forest Classifier

The final model used for classification is **Random Forest**, chosen for:

- Robustness to noise
- Ability to handle high-dimensional embeddings
- Strong performance on structured feature representations

#### Performance:
- **Training Accuracy:** ~85%
- **Testing Accuracy:** ~80%

This indicates good generalization with minimal overfitting.

---

## 🌐 Web Application (Flask)

MailShield includes a **Flask-based frontend** that allows users to test email content interactively.

### Key Features:
- Clean and modern UI (Glassmorphism design)
- Paste email content and detect spam instantly
- No page refresh (AJAX-based prediction)
- Displays prediction and confidence score
- Backend integrated with trained ML pipeline

---

## 🛠️ Tech Stack

- **Python**
- **Flask**
- **Scikit-learn**
- **Gensim (Word2Vec)**
- **NLTK / spaCy (Lemmatization & NLP)**
- **HTML / CSS / JavaScript**
- **Git LFS** (for large model files)

---

## 📁 Project Structure

