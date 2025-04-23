import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqcolbert_src.reranking.colbert_batch_reranker import colBERTReRankerBatch
 
embedding_model_path = "../model/colbert"
# embedding_model_path = "google-bert/bert-base-german-cased"
batchreranker = colBERTReRankerBatch(model_name_or_path=embedding_model_path, device='cpu')
 
query = "Was ist die Hauptstadt von Frankreich?"
 
doc_candidates = [
    "Paris ist die Hauptstadt von Frankreich.",
    "Berlin ist die Hauptstadt von Deutschland.",
    "Frankreich hat viele schöne Städte."
]
candidate_idx = list(range(len(doc_candidates)))
 
top_scores, top_indices = batchreranker.reranker(query, doc_candidates, candidate_idx, batch_size=1, top_n=2)
print("Top Scores:", top_scores)
print("Top Indices:", top_indices)
for id in top_indices:
    print(doc_candidates[id])
