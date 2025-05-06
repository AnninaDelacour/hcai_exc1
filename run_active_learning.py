import pandas as pd
from main_active_learning import active_learning_main

df = pd.read_csv("YoutubeCommentsDataSet.csv")
df = df.dropna(subset=['Comment'])


# Define the label mapping
label_map = {
    "positive": 0,
    "neutral": 1,
    "negative": 2
}

# start the active learning
active_learning_main(df, label_map=label_map)