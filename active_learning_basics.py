#!/usr/bin/env python

"""INTRODUCTION TO ACTIVE LEARNING

A simple text classification algorithm for SGDClassifier.

This code was adjusted to a specific exercise and originates from the open source example which accompanys Chapter 2 from the book:
"Human-in-the-Loop Machine Learning".

This code tries to classify the sentiment of Youtube comments into one of three categories:
  - positiv
  - neutral
  - negative

It looks for low confidence items and outliers humans should review.

"""

__original_author__ = "Robert Munro"
__license__ = "MIT"
__version__ = "1.0.1"


import random
import math
import datetime
import csv
import re
import os
from random import shuffle
from collections import defaultdict
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split




# --- SETTINGS --- #

label_map = {
    "positive": 0,
    "neutral": 1,
    "negative": 2
}


data = []
already_labeled = {} # tracking what is already labeled


def load_data_from_df(df, already_labeled_ids=None):
    data = []
    for i, row in df.iterrows():
        textid = str(i)
        if already_labeled_ids and textid in already_labeled_ids:
            continue
        text = row['Comment']
        label = row.get('Sentiment', "")
        if label_map and isinstance(label, str):
            label = label_map.get(label.lower(), "")
        sampling = row.get('strategy', "")
        conf = row.get('confidence', 0)
        data.append([textid, text, label, sampling, conf])
    return data



# --- LOAD ALL UNLABELED, TRAINING, VALIDATION, AND EVALUATION DATA --- #

annotation_instructions = "Please type 1 if the sentiment of this comment is positive, neutral, or negative, "
annotation_instructions += "or hit Enter if not.\n"
annotation_instructions += "Type 2 to go back to the last message, "
annotation_instructions += "type d to see detailed definitions, "
annotation_instructions += "or type s to save your annotations.\n"

last_instruction = "All done!\n"
last_instruction += "Type 2 to go back to change any labels,\n"
last_instruction += "or Enter to save your annotations."

detailed_instructions = "The sentiment of a comment can be interpreted in different ways.\n"
detailed_instructions += "A comment could be written:\n"
detailed_instructions += "  - in good faith, humorus or to express overall non-aggressive emotions.\n"
detailed_instructions += "  - in a sarcastic or cynical manner, or passive-aggressive, often as a subtle attack.\n"
detailed_instructions += "  - openly written in bad faith, to express pain, to hurt or to belittle others.\n"
detailed_instructions += "  - as a neutral response to someone or something, talking objectively about something or someone.\n"


def get_annotations(data, default_sampling_strategy="random"):
    """Prompts annotator for label from command line and adds annotations to data 
    
    Keyword arguments:
        data -- an list of unlabeled items where each item is 
                [ID, TEXT, LABEL, SAMPLING_STRATEGY, CONFIDENCE]
        default_sampling_strategy -- strategy to use for each item if not already specified
    """

    ind = 0
    label_options = {"positive": 0, "neutral": 1, "negative": 2}

    while ind < len(data):
        textid, text, label, strategy, confidence = data[ind]

        if textid in already_labeled:
            print(f"Skipping seen: {label}")
            ind += 1
            continue

        print(annotation_instructions)
        user_input = input(f"\n{text}\n\n> ").strip().lower()

        if user_input == "2":
            ind = max(0, ind - 1)
        elif user_input == "d":
            print(detailed_instructions)
        elif user_input == "s":
            break
        elif user_input in label_options:
            data[ind][2] = str(label_options[user_input])  # map and store as string
            already_labeled[textid] = label_options[user_input]
            if not data[ind][3]:
                data[ind][3] = default_sampling_strategy
            ind += 1
        else:
            print("Invalid input! Please enter one of: positive, neutral, negative, or [s/2/d].")

    return data


# --- TRAINING AND EVALUATION OF THE MODEL --- #

def train_model(training_data, evaluation_data):
    # Extract text and label from training data
    X_train = [row[1] for row in training_data]
    y_train = [int(row[2]) for row in training_data]  # Map Sentiment to int

    # Evaluation Data
    X_eval = [row[1] for row in evaluation_data]
    y_eval = [int(row[2]) for row in evaluation_data]

    # Vectorizing the data for SGD Classification via TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_eval_tfidf = vectorizer.transform(X_eval)

    # Classifier definition: SGD = Linear SVM)
    model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_eval_tfidf)
    f1 = f1_score(y_eval, y_pred, average='macro')

    print(classification_report(y_eval, y_pred))
    print(f"F1 Score: {f1:.3f}")

    return model, vectorizer



def get_low_conf_unlabeled(model, vectorizer, unlabeled_data, number=80, limit=10000):
    """
    Chooses those examples with the smallest prediction confidence = Uncertainty Sampling -> Margin Sampling
    """
    if limit != -1:
        unlabeled_data = random.sample(unlabeled_data, min(limit, len(unlabeled_data)))

    confidences = []

    for item in unlabeled_data:
        text = item[1]  # item = [id, text, label, strategy, confidence]
        X = vectorizer.transform([text])
        try:
            decision = model.decision_function(X)
            if decision.ndim == 1:  # binary
                confidence = abs(decision[0])
            else:  # multiclass
                confidence = np.max(decision) - np.partition(decision[0], -2)[-2]
        except Exception as e:
            continue  # Continue even after error

        item[3] = "low_confidence"
        item[4] = confidence
        confidences.append(item)

    confidences.sort(key=lambda x: x[4])  # smallest confidence level first
    return confidences[:number]


def get_random_items(unlabeled_data, number=10):
    """
    Draw random examples = Random Sampling
    """
    random.shuffle(unlabeled_data)
    selected = []
    for item in unlabeled_data:
        if item[0] in already_labeled:
            continue
        item[3] = "random"
        selected.append(item)
        if len(selected) == number:
            break
    return selected


def get_outliers(training_data, unlabeled_data, vectorizer, number=10):
    """
    Chooses examples which are least similar to the trainings set = Diversity Sampling
    """
    train_texts = [item[1] for item in training_data]
    train_vecs = vectorizer.transform(train_texts)

    outliers = []
    for item in unlabeled_data:
        if item[0] in already_labeled:
            continue
        text_vec = vectorizer.transform([item[1]])
        dist = pairwise_distances(text_vec, train_vecs, metric="cosine")
        avg_dist = dist.mean()
        item[3] = "outlier"
        item[4] = avg_dist
        outliers.append(item)

    outliers.sort(key=lambda x: -x[4])  # greatest distance first
    return outliers[:number]
    


def evaluate_model(model, vectorizer, evaluation_data, label_map):
    texts = [item[1] for item in evaluation_data]
    true_labels = [int(item[2]) for item in evaluation_data]

    X_eval = vectorizer.transform(texts)
    y_pred = model.predict(X_eval)

    f1 = f1_score(true_labels, y_pred, average='weighted')

    # One-vs-rest Binarization of labels for ROC AUC
    y_true_bin = label_binarize(true_labels, classes=list(label_map.values()))
    try:
        y_scores = model.decision_function(X_eval)
        if len(label_map) == 2 and len(y_scores.shape) == 1:
            y_scores = np.vstack([-y_scores, y_scores]).T
        auc = roc_auc_score(y_true_bin, y_scores, average='weighted', multi_class='ovr')
    except Exception as e:
        print(f"Warning: ROC AUC could not be computed ({e})")
        auc = float('nan')

    return round(f1, 3), round(auc, 3)
