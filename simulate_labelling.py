from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import label_binarize
import joblib
import numpy as np
import pandas as pd
import random



label_map = {
    "positive": 0,
    "neutral": 1,
    "negative": 2
    }



def sim_load_data_from_df(df, already_labeled_ids=None, require_label=False):
    data = []
    for i, row in df.iterrows():
        textid = str(i)
        if already_labeled_ids and textid in already_labeled_ids:
            continue
        text = row['Comment']
        label = row.get('Sentiment', "")
        if isinstance(label, str) and label_map:
            label = label_map.get(label.lower(), "")
        elif isinstance(label, (int, np.integer)):
            pass  # already numeric label
        else:
            label = ""
        if require_label and label == "":
            continue  # Skip if label required and missing
        sampling = row.get('strategy', "")
        conf = row.get('confidence', 0)
        data.append([textid, text, label, sampling, conf])
    return data



def sim_train_model(training_data, evaluation_data):
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


    

def sim_get_low_conf_unlabeled(model, vectorizer, unlabeled_data, number=80, limit=10000):
    
    # Chooses those examples with the smallest prediction confidence = Uncertainty Sampling -> Margin Sampling
    
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




'''
This is a copy from the main_active_learning.py file.
In this version we simulate the manual labelling by using the Ground Truth.
'''

def sim_active_learning_main(df, label_map, max_iterations=20, initial_label_size=20, eval_size=0.2, query_size=30):
    """
    Active Learning Routine for SGDClassifier
    """
    # 1. Split in initial labeled, evaluation, and unlabeled
    df_labeled, df_pool = train_test_split(df, train_size=initial_label_size, stratify=df['Sentiment'], random_state=42)
    df_eval, df_unlabeled = train_test_split(df_pool, test_size=eval_size, stratify=df_pool['Sentiment'], random_state=42)

    already_labeled_ids = set(map(str, df_labeled.index))
    
    for iteration in range(max_iterations):
        print(f"\n================ ACTIVE LEARNING ROUND {iteration+1} ================")

        # 2. Preprocess data
        training_data = sim_load_data_from_df(df_labeled, require_label=True)
        evaluation_data = sim_load_data_from_df(df_eval)
        unlabeled_data = sim_load_data_from_df(df_unlabeled, already_labeled_ids=already_labeled_ids)

        # 3. Train model
        model, vectorizer = sim_train_model(training_data, evaluation_data)

        # 4. Evaluate
        f1, auc = sim_evaluate_model(model, vectorizer, evaluation_data, label_map)
        print(f"[Iteration {iteration+1}] F1 Score: {f1} | AUC: {auc}")

        # 5. Use Uncertainty Sampling
        queried = sim_get_low_conf_unlabeled(model, vectorizer, unlabeled_data, number=query_size)

        # 6. Simulating manual labelling: Using ground truth as labels
        for item in queried:
            true_label = df.loc[int(item[0]), 'Sentiment']
            item[2] = label_map[true_label]
            already_labeled_ids.add(item[0])

        # 7. Add the new labelled data
        new_rows = pd.DataFrame(queried, columns=['id', 'Comment', 'Sentiment', 'strategy', 'confidence'])
        df_labeled = pd.concat([df_labeled, new_rows.set_index('id')])

        print(f"[DEBUG] df_unlabeled size before drop: {len(df_unlabeled)}")
        queried_ids = [int(item[0]) for item in queried]
        df_unlabeled = df_unlabeled.drop(index=queried_ids, errors='ignore')
        print(f"[DEBUG] df_unlabeled size after drop: {len(df_unlabeled)}")


        # 8. Optional: Abort if there is no more unlabelled data
        if len(df_unlabeled) == 0 or len(queried) == 0:
            print("No more data to annotate.")
            break

    
    joblib.dump(model, "simulating_active_learning_model_margin_sampling.pkl")
    joblib.dump(vectorizer, "simualting_active_learning_vectorizer_margin_sampling.pkl")
    print("Successfully saved model and vectorizer.")
    
    print("\nActive Learning finished successfully!")



def sim_evaluate_model(model, vectorizer, evaluation_data, label_map):
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