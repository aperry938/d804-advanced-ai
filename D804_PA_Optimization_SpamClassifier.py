"""
D804_PA_Optimization_SpamClassifier.py

Goal: Optimize the Naive Bayes Spam Classifier for improved efficiency
      and performance through systematic hyperparameter tuning and
      feature engineering.

Optimization Strategies:
1. Smoothing Parameter (Alpha) Tuning
2. Feature Selection: Vocabulary size, N-gram range
3. Vectorization Method Comparison: Count vs TF-IDF
4. Performance Benchmarks: Accuracy, Precision, Recall, F1, ROC-AUC

Author: Student
Course: D804 - Advanced AI for Computer Scientists
"""

import numpy as np
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import time
import json

# ============================================================================
# SECTION 1: BASELINE CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Baseline configuration
BASELINE_CONFIG = {
    'vectorizer': 'tfidf',
    'max_features': 5000,
    'ngram_range': (1, 2),
    'alpha': 1.0,
    'stop_words': 'english'
}

# ============================================================================
# SECTION 2: DATA LOADING AND PREPARATION
# ============================================================================

def load_dataset():
    """Load the SMS Spam Collection dataset."""
    print("Loading SMS Spam Collection dataset...")

    # Sample data for demonstration
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
spam	SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsCs apply Reply HL 4 info
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
spam	Todays Voda numbers ending 7548 are selected to receive a 350 award. If you have a match please call 08712300220 quoting claim code 7548 T&Cs apply
ham	I wanna come and hang out with you
ham	Just woken up. Yeesh. How did the exam go?
spam	Call Germany for only 1p/min from your mob! No contract, just great rates. 0800 500 1011 for info
ham	Have a good day at work?
ham	Good afternoon! How was your lunch?
spam	Win a Nintendo Wii! Complete the entry at txtauction.co.uk. Reply START. T&Cs txtauction.co.uk 18+ Cost 1.50 pm.
ham	Did you catch the bus?
ham	Aight, I'll see you at 7
spam	PRIVATE! Your 2003 Account Statement shows 800 un-claimed prize, to receive it immediately call 09066612661
ham	Thanks for lunch yesterday! What are your plans for the weekend?
ham	Be there in 5
spam	HOT LIVE CHAT! Call the BABESTATION 09099725823 and meet new hot babes! Only 60p/min. 18+
ham	I'm at the station now
ham	Sure, pick me up at 6"""

    lines = sample_data.strip().split('\n')
    data = []
    for line in lines:
        parts = line.split('\t', 1)
        if len(parts) == 2:
            data.append({'label': parts[0], 'message': parts[1]})

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")

    return df


def preprocess_text(text):
    """Clean and preprocess text."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text


def prepare_data(df):
    """Prepare data for modeling."""
    df['cleaned_message'] = df['message'].apply(preprocess_text)
    df['label_encoded'] = (df['label'] == 'spam').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_message'], df['label_encoded'],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['label_encoded']
    )

    print(f"Training: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# ============================================================================
# SECTION 3: MODEL TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, config):
    """
    Train and evaluate a Naive Bayes model with given configuration.

    Args:
        config: Dictionary with vectorizer settings and model hyperparameters

    Returns:
        Dictionary with performance metrics
    """
    start_time = time.time()

    # Create vectorizer
    if config['vectorizer'] == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            stop_words=config.get('stop_words', 'english')
        )
    else:  # count
        vectorizer = CountVectorizer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            stop_words=config.get('stop_words', 'english')
        )

    # Fit and transform
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = MultinomialNB(alpha=config['alpha'])
    model.fit(X_train_vec, y_train)

    training_time = time.time() - start_time

    # Predictions
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'training_time': training_time,
        'num_features': X_train_vec.shape[1]
    }

    return metrics, model, vectorizer

# ============================================================================
# SECTION 4: OPTIMIZATION EXPERIMENTS
# ============================================================================

def run_baseline(X_train, X_test, y_train, y_test):
    """Run baseline model."""
    print("\n" + "=" * 60)
    print("BASELINE MODEL EVALUATION")
    print("=" * 60)
    print(f"\nBaseline Configuration:")
    for key, value in BASELINE_CONFIG.items():
        print(f"  {key}: {value}")

    metrics, model, vectorizer = train_and_evaluate(
        X_train, X_test, y_train, y_test, BASELINE_CONFIG
    )

    print(f"\nBaseline Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Features:  {metrics['num_features']}")
    print(f"  Time:      {metrics['training_time']:.4f}s")

    return metrics


def optimize_alpha(X_train, X_test, y_train, y_test):
    """
    Optimize the Laplace smoothing parameter (alpha).

    Alpha Optimization:
    - alpha = 0: No smoothing (can cause zero probability issues)
    - alpha = 1: Standard Laplace smoothing
    - Higher alpha: More smoothing, less sensitive to rare words

    The optimal alpha balances handling unseen words while
    preserving discriminative power.
    """
    print("\n" + "=" * 60)
    print("OPTIMIZATION 1: SMOOTHING PARAMETER (ALPHA)")
    print("=" * 60)

    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = []

    for alpha in alphas:
        config = BASELINE_CONFIG.copy()
        config['alpha'] = alpha

        metrics, _, _ = train_and_evaluate(X_train, X_test, y_train, y_test, config)
        metrics['alpha'] = alpha
        results.append(metrics)

        print(f"  alpha={alpha:<6} -> Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

    best = max(results, key=lambda x: x['f1_score'])
    print(f"\nBest alpha: {best['alpha']} (F1: {best['f1_score']:.4f})")

    return results, best['alpha']


def optimize_features(X_train, X_test, y_train, y_test, best_alpha):
    """
    Optimize feature extraction parameters.

    Feature Optimization:
    - Max features: Vocabulary size (more = more info, but slower)
    - N-gram range: (1,1) unigrams, (1,2) uni+bigrams, (1,3) up to trigrams
    """
    print("\n" + "=" * 60)
    print("OPTIMIZATION 2: FEATURE EXTRACTION")
    print("=" * 60)

    feature_configs = [
        {'max_features': 1000, 'ngram_range': (1, 1)},
        {'max_features': 3000, 'ngram_range': (1, 1)},
        {'max_features': 5000, 'ngram_range': (1, 1)},
        {'max_features': 3000, 'ngram_range': (1, 2)},
        {'max_features': 5000, 'ngram_range': (1, 2)},
        {'max_features': 5000, 'ngram_range': (1, 3)},
    ]

    results = []
    for fc in feature_configs:
        config = BASELINE_CONFIG.copy()
        config.update(fc)
        config['alpha'] = best_alpha

        metrics, _, _ = train_and_evaluate(X_train, X_test, y_train, y_test, config)
        metrics['feature_config'] = fc
        results.append(metrics)

        print(f"  features={fc['max_features']}, ngrams={fc['ngram_range']} -> "
              f"F1: {metrics['f1_score']:.4f}, Features: {metrics['num_features']}")

    best = max(results, key=lambda x: x['f1_score'])
    print(f"\nBest feature config: {best['feature_config']}")

    return results, best['feature_config']


def optimize_vectorizer(X_train, X_test, y_train, y_test, best_alpha, best_features):
    """
    Compare vectorization methods.

    Vectorization Comparison:
    - CountVectorizer: Raw word counts
    - TfidfVectorizer: TF-IDF weighted counts

    TF-IDF typically works better as it down-weights common words
    and up-weights discriminative words.
    """
    print("\n" + "=" * 60)
    print("OPTIMIZATION 3: VECTORIZATION METHOD")
    print("=" * 60)

    vectorizers = ['count', 'tfidf']
    results = []

    for vec_type in vectorizers:
        config = BASELINE_CONFIG.copy()
        config.update(best_features)
        config['alpha'] = best_alpha
        config['vectorizer'] = vec_type

        metrics, _, _ = train_and_evaluate(X_train, X_test, y_train, y_test, config)
        metrics['vectorizer'] = vec_type
        results.append(metrics)

        print(f"  {vec_type:<10} -> Accuracy: {metrics['accuracy']:.4f}, "
              f"F1: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    best = max(results, key=lambda x: x['f1_score'])
    print(f"\nBest vectorizer: {best['vectorizer']}")

    return results, best['vectorizer']

# ============================================================================
# SECTION 5: FINAL OPTIMIZED MODEL
# ============================================================================

def train_optimized_model(X_train, X_test, y_train, y_test,
                          best_alpha, best_features, best_vectorizer):
    """Train the final optimized model."""
    print("\n" + "=" * 60)
    print("TRAINING OPTIMIZED MODEL")
    print("=" * 60)

    optimized_config = BASELINE_CONFIG.copy()
    optimized_config.update(best_features)
    optimized_config['alpha'] = best_alpha
    optimized_config['vectorizer'] = best_vectorizer

    print("\nOptimized Configuration:")
    for key, value in optimized_config.items():
        print(f"  {key}: {value}")

    metrics, model, vectorizer = train_and_evaluate(
        X_train, X_test, y_train, y_test, optimized_config
    )

    print(f"\nOptimized Model Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Features:  {metrics['num_features']}")
    print(f"  Time:      {metrics['training_time']:.4f}s")

    return metrics, model, vectorizer, optimized_config

# ============================================================================
# SECTION 6: BENCHMARKING AND COMPARISON
# ============================================================================

def benchmark_comparison(baseline_metrics, optimized_metrics, baseline_config, optimized_config):
    """
    Compare baseline and optimized models with defined benchmarks.

    Benchmarks for Evaluation:
    1. Accuracy: Overall classification correctness
    2. Precision: Spam detection accuracy (avoid false positives)
    3. Recall: Spam capture rate (catch all spam)
    4. F1-Score: Balanced precision-recall metric
    5. ROC-AUC: Model's discriminative ability
    6. Training Time: Computational efficiency
    """
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)

    print("\n{:<20} {:>15} {:>15} {:>15}".format(
        "Benchmark", "Baseline", "Optimized", "Improvement"
    ))
    print("-" * 65)

    benchmarks = [
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1-Score', 'f1_score'),
        ('ROC-AUC', 'roc_auc'),
        ('Training Time', 'training_time'),
        ('Num Features', 'num_features'),
    ]

    for name, key in benchmarks:
        base_val = baseline_metrics[key]
        opt_val = optimized_metrics[key]

        if key in ['training_time']:
            improvement = ((base_val - opt_val) / base_val) * 100 if base_val > 0 else 0
            print(f"{name:<20} {base_val:>15.4f} {opt_val:>15.4f} {improvement:>+14.1f}%")
        elif key == 'num_features':
            improvement = ((opt_val - base_val) / base_val) * 100 if base_val > 0 else 0
            print(f"{name:<20} {base_val:>15} {opt_val:>15} {improvement:>+14.1f}%")
        else:
            improvement = ((opt_val - base_val) / base_val) * 100 if base_val > 0 else 0
            print(f"{name:<20} {base_val:>15.4f} {opt_val:>15.4f} {improvement:>+14.1f}%")

    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)

    print("\nConfiguration Changes:")
    for key in optimized_config:
        if key in baseline_config and baseline_config[key] != optimized_config[key]:
            print(f"  {key}: {baseline_config[key]} -> {optimized_config[key]}")

    f1_improvement = ((optimized_metrics['f1_score'] - baseline_metrics['f1_score'])
                      / baseline_metrics['f1_score']) * 100 if baseline_metrics['f1_score'] > 0 else 0

    print(f"\nOverall F1-Score Improvement: {f1_improvement:+.2f}%")

    if f1_improvement > 0:
        print("Optimization successful: Model performance improved.")
    elif f1_improvement == 0:
        print("Note: Baseline configuration was already optimal for this dataset.")
    else:
        print("Note: Baseline performed better; consider reverting changes.")


def plot_optimization_results(baseline_metrics, optimized_metrics, alpha_results):
    """Visualize optimization results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Metric comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    x = np.arange(len(metrics))
    width = 0.35

    baseline_vals = [baseline_metrics[m] for m in metrics]
    optimized_vals = [optimized_metrics[m] for m in metrics]

    axes[0].bar(x - width/2, baseline_vals, width, label='Baseline', color='steelblue')
    axes[0].bar(x + width/2, optimized_vals, width, label='Optimized', color='darkorange')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Performance Metrics: Baseline vs Optimized')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'])
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', alpha=0.3)

    # Alpha tuning results
    alphas = [r['alpha'] for r in alpha_results]
    f1_scores = [r['f1_score'] for r in alpha_results]

    axes[1].plot(alphas, f1_scores, 'o-', color='steelblue', linewidth=2, markersize=8)
    axes[1].set_xlabel('Alpha (Smoothing Parameter)')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('Alpha Parameter Tuning')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    # Mark best alpha
    best_idx = np.argmax(f1_scores)
    axes[1].axvline(x=alphas[best_idx], color='red', linestyle='--', alpha=0.7)
    axes[1].annotate(f'Best: {alphas[best_idx]}',
                     xy=(alphas[best_idx], f1_scores[best_idx]),
                     xytext=(alphas[best_idx]*2, f1_scores[best_idx]-0.02),
                     fontsize=10)

    plt.tight_layout()
    plt.savefig('spam_optimization_comparison.png', dpi=150)
    print("\nOptimization comparison plot saved to 'spam_optimization_comparison.png'")

# ============================================================================
# SECTION 7: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main optimization pipeline.

    Steps:
    1. Load and prepare data
    2. Evaluate baseline model
    3. Optimize smoothing parameter (alpha)
    4. Optimize feature extraction
    5. Compare vectorization methods
    6. Train final optimized model
    7. Benchmark comparison
    """
    print("=" * 60)
    print("D804_PA_Optimization_SpamClassifier")
    print("Naive Bayes Spam Classifier Optimization")
    print("=" * 60)

    # Load and prepare data
    df = load_dataset()
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Step 1: Baseline evaluation
    baseline_metrics = run_baseline(X_train, X_test, y_train, y_test)

    # Step 2: Optimize alpha
    alpha_results, best_alpha = optimize_alpha(X_train, X_test, y_train, y_test)

    # Step 3: Optimize features
    feature_results, best_features = optimize_features(
        X_train, X_test, y_train, y_test, best_alpha
    )

    # Step 4: Compare vectorizers
    vec_results, best_vectorizer = optimize_vectorizer(
        X_train, X_test, y_train, y_test, best_alpha, best_features
    )

    # Step 5: Train optimized model
    optimized_metrics, model, vectorizer, optimized_config = train_optimized_model(
        X_train, X_test, y_train, y_test,
        best_alpha, best_features, best_vectorizer
    )

    # Step 6: Benchmark comparison
    benchmark_comparison(baseline_metrics, optimized_metrics,
                         BASELINE_CONFIG, optimized_config)

    # Step 7: Visualize
    plot_optimization_results(baseline_metrics, optimized_metrics, alpha_results)

    # Save results
    results = {
        'baseline_metrics': baseline_metrics,
        'optimized_metrics': optimized_metrics,
        'optimized_config': optimized_config,
        'alpha_tuning': alpha_results
    }

    with open('spam_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to 'spam_optimization_results.json'")

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    return baseline_metrics, optimized_metrics


if __name__ == "__main__":
    baseline_metrics, optimized_metrics = main()
