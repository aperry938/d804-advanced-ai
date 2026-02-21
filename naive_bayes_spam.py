"""
naive_bayes_spam.py

Spam Detection using Naive Bayes (Probabilistic Method)

Dataset: SMS Spam Collection from UCI Machine Learning Repository
Method: Multinomial Naive Bayes with TF-IDF Vectorization

This model classifies SMS messages as spam or ham (not spam) using
probabilistic reasoning based on Bayes' theorem. The Naive Bayes
classifier assumes conditional independence between features
given the class label.

Mathematical Foundation:
P(Spam|Message) = P(Message|Spam) * P(Spam) / P(Message)

The "naive" assumption treats each word's probability as independent:
P(Message|Spam) = P(word1|Spam) * P(word2|Spam) * ... * P(wordN|Spam)

Author: Anthony Perry
"""

import numpy as np
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import urllib.request
import os

# ============================================================================
# SECTION 1: CONFIGURATION AND PARAMETERS
# ============================================================================

# Data parameters
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_FILE = "SMSSpamCollection"
TEST_SIZE = 0.2             # 20% of data for testing
RANDOM_STATE = 42           # For reproducibility

# Vectorization parameters
MAX_FEATURES = 5000         # Maximum vocabulary size
NGRAM_RANGE = (1, 2)        # Use unigrams and bigrams

# Naive Bayes parameters
ALPHA = 1.0                 # Laplace smoothing parameter

# ============================================================================
# SECTION 2: DATA LOADING AND PREPARATION
# ============================================================================

def load_dataset():
    """
    Load the SMS Spam Collection dataset.

    Dataset Description:
    - Source: UCI Machine Learning Repository
    - Total samples: 5,574 SMS messages
    - Labels: 'ham' (legitimate) or 'spam'
    - Features: Raw text messages

    The dataset is tab-separated with format: label<tab>message

    Returns:
        pandas DataFrame with 'label' and 'message' columns
    """
    print("=" * 60)
    print("LOADING SMS SPAM COLLECTION DATASET")
    print("=" * 60)

    # Create sample data inline for demonstration
    # In production, download from UCI repository
    sample_data = """ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
ham	Ok lar... Joking wif u oni...
spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
ham	U dun say so early hor... U c already then say...
ham	Nah I don't think he goes to usf, he lives around here though
spam	FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, 1.50 to rcv
ham	Even my brother is not like to speak with me. They treat me like aids patent.
ham	As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune
spam	WINNER!! As a valued network customer you have been selected to receivea 900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
spam	Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with no extra charge! Call The Mobile Update Co FREE on 08002986030
ham	I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.
spam	SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TssCs apply Reply HL 4 info
spam	URGENT! You have won a 1 week FREE membership in our 100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18
ham	I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times.
ham	I TORTURE you then give up!!!
spam	Thanks for your subscription to Ringtone UK your mobile will be charged 5/month Please confirm by replying YES or NO. If you reply NO you will not be charged
ham	Sorry, I'll call later
ham	Wow, thats impressive. Had to use alarm to wake up. I would still be in bed if not for that alarm!!!
spam	Congrats! 1 year special cinema pass for 2 is yours. call 09061209465 now! C Suprman V, Matrix3, StarWars3, all 4 FREE! bx420telecom.net
ham	Oh k...i'm watching here:)
spam	Your free ringtone is waiting to be collected. Simply text the password 'MIX' to 85069 to verify. Get Usher and Britney. txt]
ham	Hello! How's you and how did saturday go? I was just texting to see if you'd decided to do anything tomorrow. Not that i'm trying to blag an invite or anything!
ham	Ugh my leg hurts from working out
spam	XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap.xxxmobile/telecom.net/telecom
ham	What you doing?how are you?
ham	Going to take your bro this weekend?
ham	Hello handsome! How is it going? Just got in from work about an hour ago. How is your day going?
spam	FREE for 1st week! No1 Nokia tone 4 ur mob every week just txt NOKIA to 8077 Get txting and tell ur mates
ham	I dunno until when... Lets444444444444444see how... ok?
spam	Ur going to receive 900 for your loan repayment. Please reply with YES/NO.
ham	Do you guys want anything from mcds?
ham	Where are you man?
spam	As a valued customer, you have been selected from all our mobile customers to receive a 2000 pounds prize
ham	Are you OK? No practice with family around?
ham	Yeah so wat time do we meet?
spam	Todays Voda numbers ending 7548 are selected to receive a 350 award. If you have a match please call 08712300220 quoting claim code 7548 T&C's apply
ham	I wanna come and hang out with you
ham	Just woken up. Yeesh. How did the exam go?
spam	Call Germany for only 1p/min from your mob! No contract, just great rates. 0800 500 1011 for info
ham	Have a good day at work?
ham	Good afternoon! How was your lunch?
spam	Win a Nintendo Wii! Complete the entry at txtauction.co.uk. Reply START. T&C's txtauction.co.uk 18+ Cost 1.50 pm.
ham	Did you catch the bus?
ham	Aight, I'll see you at 7
spam	PRIVATE! Your 2003 Account Statement shows 800 un-claimed prize, to receive it immediately call 09066612661
ham	Thanks for lunch yesterday! What are your plans for the weekend?
ham	Be there in 5
spam	HOT LIVE CHAT! Call the BABESTATION 09099725823 and meet new hot babes! Only 60p/min. 18+
ham	I'm at the station now
ham	Sure, pick me up at 6"""

    # Parse the sample data
    lines = sample_data.strip().split('\n')
    data = []
    for line in lines:
        parts = line.split('\t', 1)
        if len(parts) == 2:
            data.append({'label': parts[0], 'message': parts[1]})

    df = pd.DataFrame(data)

    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(f"\nSpam percentage: {(df['label'] == 'spam').mean() * 100:.2f}%")

    return df


def describe_dataset(df):
    """
    Provide detailed description of the dataset features.

    Dataset Features:
    1. Label (Target): Binary classification
       - 'ham': Legitimate messages (non-spam)
       - 'spam': Unsolicited/unwanted messages

    2. Message (Feature): Raw text content
       - Variable length SMS messages
       - Contains various linguistic patterns
       - Includes special characters, numbers, abbreviations

    Spam Characteristics (features the model learns):
    - Presence of keywords: "free", "win", "prize", "call", "txt"
    - Excessive use of capitalization
    - Presence of phone numbers and URLs
    - Urgency language: "urgent", "now", "immediately"
    - Financial references: monetary amounts, "cash"
    """
    print("\n" + "=" * 60)
    print("DATASET DESCRIPTION")
    print("=" * 60)

    print("\nFeature Analysis:")
    print("-" * 40)

    # Message length statistics
    df['message_length'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))

    print("\nMessage Length Statistics:")
    print(f"  Mean length: {df['message_length'].mean():.2f} characters")
    print(f"  Max length:  {df['message_length'].max()} characters")
    print(f"  Min length:  {df['message_length'].min()} characters")

    print("\nWord Count Statistics:")
    print(f"  Mean words: {df['word_count'].mean():.2f}")
    print(f"  Max words:  {df['word_count'].max()}")
    print(f"  Min words:  {df['word_count'].min()}")

    # Compare ham vs spam
    print("\nComparison by Class:")
    print("-" * 40)
    for label in ['ham', 'spam']:
        subset = df[df['label'] == label]
        print(f"\n{label.upper()} Messages:")
        print(f"  Count: {len(subset)}")
        print(f"  Avg length: {subset['message_length'].mean():.2f} chars")
        print(f"  Avg words: {subset['word_count'].mean():.2f}")

    # Sample messages
    print("\nSample Messages:")
    print("-" * 40)
    print("\nHam example:")
    print(f"  '{df[df['label'] == 'ham']['message'].iloc[0]}'")
    print("\nSpam example:")
    print(f"  '{df[df['label'] == 'spam']['message'].iloc[0]}'")

    return df


def preprocess_text(text):
    """
    Clean and preprocess text for modeling.

    Data Preparation Techniques:
    1. Lowercase conversion: Normalizes case (FREE == free)
    2. Remove punctuation: Eliminates special characters
    3. Remove numbers: Reduces noise from phone numbers, codes
    4. Remove extra whitespace: Standardizes spacing

    These techniques ensure consistent input for the vectorizer
    and reduce vocabulary size while preserving semantic meaning.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def prepare_data(df):
    """
    Prepare the dataset for model training.

    Data Preparation Steps:
    1. Text preprocessing: Clean all messages
    2. Label encoding: Convert 'ham'/'spam' to 0/1
    3. Train/test split: Stratified split to maintain class balance
    4. TF-IDF Vectorization: Convert text to numerical features

    TF-IDF (Term Frequency-Inverse Document Frequency):
    - Term Frequency: How often a word appears in a document
    - Inverse Document Frequency: Log of (total docs / docs containing word)
    - TF-IDF = TF * IDF
    - Words common in one document but rare overall get higher weights

    Args:
        df: DataFrame with 'label' and 'message' columns

    Returns:
        X_train, X_test: TF-IDF feature matrices
        y_train, y_test: Binary labels
        vectorizer: Fitted TF-IDF vectorizer
    """
    print("\n" + "=" * 60)
    print("PREPARING DATA FOR MODELING")
    print("=" * 60)

    # Step 1: Preprocess text
    print("\nStep 1: Preprocessing text...")
    df['cleaned_message'] = df['message'].apply(preprocess_text)

    # Step 2: Encode labels
    print("Step 2: Encoding labels (ham=0, spam=1)...")
    df['label_encoded'] = (df['label'] == 'spam').astype(int)

    # Step 3: Split data
    print(f"Step 3: Splitting data ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} train/test)...")
    X = df['cleaned_message']
    y = df['label_encoded']

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class balance
    )

    # Step 4: TF-IDF Vectorization
    print("Step 4: Applying TF-IDF vectorization...")
    print(f"  Max features: {MAX_FEATURES}")
    print(f"  N-gram range: {NGRAM_RANGE}")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words='english'
    )

    # Fit on training data only, transform both
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print(f"\nData preparation complete!")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Feature dimension: {X_train.shape[1]}")
    print(f"  Training spam ratio: {y_train.mean():.2%}")
    print(f"  Test spam ratio: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, vectorizer

# ============================================================================
# SECTION 3: NAIVE BAYES MODEL DEVELOPMENT
# ============================================================================

def build_naive_bayes_model():
    """
    Build the Multinomial Naive Bayes classifier.

    Naive Bayes Method Explanation:
    ================================

    Bayes' Theorem:
    P(Class|Features) = P(Features|Class) * P(Class) / P(Features)

    For spam classification:
    P(Spam|Words) = P(Words|Spam) * P(Spam) / P(Words)

    The "Naive" Assumption:
    Features (words) are conditionally independent given the class.
    P(word1, word2, ..., wordN | Spam) = P(word1|Spam) * P(word2|Spam) * ... * P(wordN|Spam)

    Multinomial Naive Bayes:
    - Suitable for text classification with word counts/frequencies
    - Models the probability of words appearing in each class
    - Uses Laplace smoothing to handle unseen words

    Laplace Smoothing (alpha parameter):
    - Adds alpha to all word counts to prevent zero probabilities
    - P(word|class) = (count(word, class) + alpha) / (total_words_in_class + alpha * vocab_size)
    - alpha=1 is standard Laplace smoothing

    Returns:
        MultinomialNB classifier (unfitted)
    """
    print("\n" + "=" * 60)
    print("BUILDING NAIVE BAYES MODEL")
    print("=" * 60)

    print("\nModel: Multinomial Naive Bayes")
    print(f"Smoothing parameter (alpha): {ALPHA}")

    print("\nProbabilistic Foundation:")
    print("  P(Spam|Message) ∝ P(Message|Spam) × P(Spam)")
    print("  P(Ham|Message) ∝ P(Message|Ham) × P(Ham)")
    print("\n  Classification: argmax[P(Class|Message)]")

    print("\nNaive Independence Assumption:")
    print("  P(Message|Class) = Π P(word_i|Class)")
    print("  Each word's probability is independent given the class")

    model = MultinomialNB(alpha=ALPHA)

    return model

# ============================================================================
# SECTION 4: MODEL TRAINING
# ============================================================================

def train_model(model, X_train, y_train):
    """
    Train the Naive Bayes model.

    Training Process:
    1. Calculate prior probabilities: P(Spam), P(Ham)
    2. Calculate likelihood for each word given each class
    3. Apply Laplace smoothing to handle zero counts

    The training is extremely fast for Naive Bayes because it only
    requires counting word occurrences per class and computing
    probabilities - no iterative optimization needed.

    Args:
        model: MultinomialNB classifier
        X_train: TF-IDF training features
        y_train: Training labels

    Returns:
        Fitted model
    """
    print("\n" + "=" * 60)
    print("TRAINING NAIVE BAYES MODEL")
    print("=" * 60)

    print("\nTraining the model...")
    model.fit(X_train, y_train)

    # Display learned parameters
    print("\nLearned Parameters:")
    print(f"  Class prior (log) - Ham:  {model.class_log_prior_[0]:.4f}")
    print(f"  Class prior (log) - Spam: {model.class_log_prior_[1]:.4f}")
    print(f"  P(Ham):  {np.exp(model.class_log_prior_[0]):.4f}")
    print(f"  P(Spam): {np.exp(model.class_log_prior_[1]):.4f}")
    print(f"\n  Number of features: {model.n_features_in_}")

    print("\nModel training complete!")

    return model

# ============================================================================
# SECTION 5: MODEL EVALUATION AND VALIDATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained Naive Bayes model.

    Validation Metrics:
    1. Accuracy: Overall correct predictions
    2. Precision: Of predicted spam, how many are actually spam
       (Important to avoid marking legitimate messages as spam)
    3. Recall: Of actual spam, how many did we catch
       (Important to filter out unwanted messages)
    4. F1-Score: Harmonic mean balancing precision and recall
    5. ROC-AUC: Area under the ROC curve (discrimination ability)
    6. Confusion Matrix: Detailed breakdown of predictions

    These metrics validate the model's probabilistic predictions
    and ensure reliable spam detection.

    Args:
        model: Trained MultinomialNB model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary containing all metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATING MODEL PERFORMANCE")
    print("=" * 60)

    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Print results
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Ham    Spam")
    print(f"  Actual Ham   {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"  Actual Spam  {cm[1][0]:5d}   {cm[1][1]:5d}")

    # Interpretation
    print("\nInterpretation:")
    print(f"  True Negatives (Ham correctly identified):  {cm[0][0]}")
    print(f"  False Positives (Ham misclassified as Spam): {cm[0][1]}")
    print(f"  False Negatives (Spam missed): {cm[1][0]}")
    print(f"  True Positives (Spam correctly caught): {cm[1][1]}")

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def plot_results(y_test, metrics):
    """
    Visualize model performance with ROC curve and confusion matrix.

    Args:
        y_test: True labels
        metrics: Dictionary with predictions and probabilities
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    axes[0].plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve - Spam Detection')
    axes[0].legend()
    axes[0].grid(True)

    # Confusion Matrix Heatmap
    cm = metrics['confusion_matrix']
    im = axes[1].imshow(cm, cmap='Blues')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Ham', 'Spam'])
    axes[1].set_yticklabels(['Ham', 'Spam'])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Confusion Matrix')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, cm[i, j], ha='center', va='center',
                        fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig('spam_classifier_results.png', dpi=150)
    print("\nResults plot saved to 'spam_classifier_results.png'")

# ============================================================================
# SECTION 6: FEATURE ANALYSIS
# ============================================================================

def analyze_features(model, vectorizer):
    """
    Analyze which words are most indicative of spam vs ham.

    This demonstrates the probabilistic nature of Naive Bayes
    by showing the log-probabilities learned for each word.
    """
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)

    feature_names = vectorizer.get_feature_names_out()

    # Get log probabilities for each class
    ham_log_probs = model.feature_log_prob_[0]
    spam_log_probs = model.feature_log_prob_[1]

    # Calculate log-odds ratio (spam vs ham)
    log_odds = spam_log_probs - ham_log_probs

    # Get top spam indicators
    spam_indices = np.argsort(log_odds)[-15:][::-1]
    print("\nTop 15 Spam Indicators (highest P(word|spam)/P(word|ham)):")
    print("-" * 50)
    for idx in spam_indices:
        print(f"  {feature_names[idx]:20s} log-odds: {log_odds[idx]:.4f}")

    # Get top ham indicators
    ham_indices = np.argsort(log_odds)[:15]
    print("\nTop 15 Ham Indicators (highest P(word|ham)/P(word|spam)):")
    print("-" * 50)
    for idx in ham_indices:
        print(f"  {feature_names[idx]:20s} log-odds: {log_odds[idx]:.4f}")

# ============================================================================
# SECTION 7: DEPLOYMENT CONSIDERATIONS
# ============================================================================

def demonstrate_deployment(model, vectorizer):
    """
    Demonstrate how the model could be deployed for end users.

    Deployment Options:
    1. REST API: Flask/FastAPI endpoint for real-time classification
    2. Email/SMS Gateway Integration: Filter messages at server level
    3. Mobile App: On-device classification using scikit-learn
    4. Batch Processing: Process message logs in bulk

    The Naive Bayes model is lightweight and fast, making it
    suitable for real-time deployment scenarios.
    """
    print("\n" + "=" * 60)
    print("DEPLOYMENT CONSIDERATIONS")
    print("=" * 60)

    # Save the model
    import joblib
    joblib.dump(model, 'spam_classifier_model.joblib')
    joblib.dump(vectorizer, 'spam_classifier_vectorizer.joblib')
    print("\nModel saved: 'spam_classifier_model.joblib'")
    print("Vectorizer saved: 'spam_classifier_vectorizer.joblib'")

    print("\nDeployment Options:")
    print("\n1. REST API Deployment:")
    print("   - Create Flask/FastAPI endpoint")
    print("   - Accept text messages via POST request")
    print("   - Return spam probability and classification")
    print("   - Example response: {'spam_probability': 0.95, 'is_spam': True}")

    print("\n2. Email/SMS Gateway Integration:")
    print("   - Integrate with email servers (postfix, sendmail)")
    print("   - Filter messages before delivery to inbox")
    print("   - Move spam to separate folder or reject")

    print("\n3. Mobile Application:")
    print("   - Export model to ONNX format for cross-platform")
    print("   - On-device classification for privacy")
    print("   - No internet required after model download")

    print("\n4. Batch Processing:")
    print("   - Process historical message logs")
    print("   - Generate spam reports and analytics")
    print("   - Train on new data periodically")

    # Demonstrate prediction
    print("\n" + "-" * 40)
    print("Live Prediction Demo:")
    print("-" * 40)

    test_messages = [
        "Hey, are you free for lunch tomorrow?",
        "CONGRATULATIONS! You won $1000 cash prize! Call now to claim!",
        "Can you pick up some milk on your way home?",
        "FREE entry to win tickets! Text WIN to 12345"
    ]

    for msg in test_messages:
        cleaned = preprocess_text(msg)
        features = vectorizer.transform([cleaned])
        proba = model.predict_proba(features)[0]
        pred = "SPAM" if proba[1] > 0.5 else "HAM"
        print(f"\nMessage: '{msg[:50]}...'")
        print(f"  Prediction: {pred} (spam probability: {proba[1]:.2%})")

# ============================================================================
# SECTION 8: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for the spam classifier pipeline.

    Pipeline Steps:
    1. Load the SMS Spam Collection dataset
    2. Describe dataset features
    3. Prepare data (preprocess, vectorize, split)
    4. Build Naive Bayes model
    5. Train the model
    6. Evaluate performance
    7. Analyze learned features
    8. Demonstrate deployment
    """
    print("=" * 60)
    print("Naive Bayes Spam Classifier")
    print("Naive Bayes Spam Detection")
    print("=" * 60)

    # Step 1: Load dataset
    df = load_dataset()

    # Step 2: Describe features
    df = describe_dataset(df)

    # Step 3: Prepare data
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df)

    # Step 4: Build model
    model = build_naive_bayes_model()

    # Step 5: Train model
    model = train_model(model, X_train, y_train)

    # Step 6: Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Step 7: Plot results
    plot_results(y_test, metrics)

    # Step 8: Analyze features
    analyze_features(model, vectorizer)

    # Step 9: Demonstrate deployment
    demonstrate_deployment(model, vectorizer)

    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)

    return model, vectorizer, metrics


if __name__ == "__main__":
    model, vectorizer, metrics = main()
