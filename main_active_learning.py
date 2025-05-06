import pandas as pd
from sklearn.model_selection import train_test_split
from active_learning_basics import (
    load_data_from_df,
    train_model,
    evaluate_model,
    get_low_conf_unlabeled
)
from labelstudio_interface import upload_to_labelstudio, fetch_labeled_data
import joblib



def active_learning_main(df, label_map, max_iterations=5, initial_label_size=20, eval_size=0.2, query_size=30):
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
        training_data = load_data_from_df(df_labeled)
        evaluation_data = load_data_from_df(df_eval)
        unlabeled_data = load_data_from_df(df_unlabeled, already_labeled_ids=already_labeled_ids)

        # 3. Train model
        model, vectorizer = train_model(training_data, evaluation_data)

        # 4. Evaluate
        f1, auc = evaluate_model(model, vectorizer, evaluation_data, label_map)
        print(f"[Iteration {iteration+1}] F1 Score: {f1} | AUC: {auc}")

        # 5. Use Uncertainty Sampling
        queried = get_low_conf_unlabeled(model, vectorizer, unlabeled_data, number=query_size)

        # 6. (Optional) Simulating manual labelling: Using ground truth as labels
        #for item in queried:
        #    true_label = df.loc[int(item[0]), 'Sentiment']
        #    item[2] = str(label_map[true_label])  # Mapping to integer-value representation
        #    already_labeled_ids.add(item[0])

        # 6. Human annotation step in Label Studio
        upload_to_labelstudio(queried)
        input("Data sent to Label Studio. Annotate data in Label Studio and click enter after you're done.")
        queried = fetch_labeled_data(queried)

        # 7. Add the new labelled data
        new_rows = pd.DataFrame(queried, columns=['id', 'Comment', 'Sentiment', 'strategy', 'confidence'])
        df_labeled = pd.concat([df_labeled, new_rows.set_index('id')])

        # 8. Optional: Abort if there is no more unlabelled data
        if len(df_unlabeled) == 0 or len(queried) == 0:
            print("No more data to annotate.")
            break

    
    joblib.dump(model, "final_active_learning_model.pkl")
    joblib.dump(vectorizer, "final_vectorizer.pkl")
    print("Successfully saved model and vectorizer.")
    
    print("\nActive Learning finished successfully!")
