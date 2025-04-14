import pandas as pd
from test_cb_tdqq.choosing_negatives.choosing_hard_negatives import build_hard_triplets

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
build_hard_triplets(
    df=df,
    model_name_or_path='all-MiniLM-L6-v2',
    start_index=2,
    n_negatives=2,
    triplet_output_path='hard_triplets.csv',
    index_name='hard.index',
    index_output_path='./indices_test',
    use_gpu=False
)

triplet_df = pd.read_csv("hard_triplets.csv")

# Step 4: Load and show results
print("Sample triplets:")
print(triplet_df)