import pandas as pd
from test_cb_tdqq.choosing_negatives.choosing_random_negatives import build_random_triplets

# Step 1: Sample DataFrame
data = {
    "sentence": [
        "A man is playing guitar.",
        "A woman is cooking.",
        "Children are playing soccer.",
        "A cat is sleeping on the couch.",
        "He is reading a book."
    ],
    "positive_sentence": [
        "Someone plays a musical instrument.",
        "She is preparing food.",
        "Kids are having fun outdoors.",
        "An animal rests on furniture.",
        "A person is reading."
    ]
}

df = pd.DataFrame(data)

# Step 3: Run it and save to CSV
output_path = "triplets_example.csv"
build_random_triplets(
    df=df,
    n_negatives=2,
    triplet_output_path=output_path
)

triplet_df = pd.read_csv("triplets_example.csv")

# Step 4: Load and show results
print("Sample triplets:")
print(triplet_df.head())