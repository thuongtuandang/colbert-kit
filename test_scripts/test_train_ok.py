import pandas as pd
import torch
from tqcolbert_src.training import ColBERTTrainer

# Create simple test dataset
test_data = [
    ("A man is playing a guitar.", "A person is playing a musical instrument.", "A bird is flying in the sky."),
    ("A woman is cooking food.", "A person is preparing a meal.", "A car is driving down the road."),
    ("A child is reading a book.", "A young person is engaged with a novel.", "The sun is setting behind the mountains.")
]
test_df = pd.DataFrame(test_data, columns=["sentence", "positive_sentence", "negative_sentence"])
test_df.to_csv("../data_test/test_triplets.csv", sep='\t', index=False)

# Load BERT model for testing
bert_model_name = "../model/colbert"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainer = ColBERTTrainer(
    model_name_or_path=bert_model_name,
    device=device,
    triplets_path="../data_test/test_triplets.csv",
    batch_size=2,
    lr=2e-5,
    num_epochs=4,
    checkpoint_dir="../test_checkpoints/"
)
trainer.train()
trainer.save_model("../model/saved_model_test/")

trainer = ColBERTTrainer(
    model_name_or_path=bert_model_name,
    # triplets=test_data,
    triplets_path="../data_test/test_triplets.csv",
    device = device,
    batch_size=256,
    lr=2e-5,
    num_epochs=3,
    checkpoint_dir="../test_checkpoints/",
    checkpoint_path="../test_checkpoints/checkpoint_epoch_2.pth"
)
trainer.train()
trainer.save_model("../model/saved_model_test_3/")