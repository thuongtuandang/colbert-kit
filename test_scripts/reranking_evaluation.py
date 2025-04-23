import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from colbert_kit_src.reranking.colbert_batch_reranker import colBERTReRankerBatch
from colbert_kit_src.indexing.bm25.bm_25 import BM25Corpus
from colbert_kit_src.retrieving.bm25_search import BM25_search
from colbert_kit_src.evaluating.ir_eval import IREvaluation
import pandas as pd
from tqdm import tqdm

eval_name = 'miracl-de-dev'
print("Test dataset: ", eval_name)
eval_data_path = '../data_test/' + f'{eval_name}/'
query_path = eval_data_path + 'queries.csv'
doc_path = eval_data_path + 'docs.csv'
qrel_path = eval_data_path + 'qrels.csv'

text_column = 'text'

df_queries = pd.read_csv(query_path)
df_qrels = pd.read_csv(qrel_path)[:10]
df_docs = pd.read_csv(doc_path)[:30]

df_docs['index'] = df_docs.index
doc_ids = df_docs.set_index('index')['doc_id'].to_dict()
texts = df_docs[text_column].tolist()

# Create BM25 corpus
bm25_corpus = BM25Corpus(texts, language='de')
# Tokenize texts, remove stopwords, etc
corpus = bm25_corpus.clean_token()
# Create a corpus
bm25 = bm25_corpus.create_corpus()

"""Process queries and perform IR evaluation."""

# Load BM25 index for retrieval
BM25search = BM25_search(bm25=bm25, language='de')

results = []
query_ids = df_qrels['query_id'].unique()
top_k = 100

# Load colBERT
model_name_or_path = "thuongtuandang/german-colbert"

colbert_reranker_batch = colBERTReRankerBatch(model_name_or_path=model_name_or_path, device="cpu")

for query_id in tqdm(query_ids, desc="Processing queries"):
    query_row = df_queries.loc[df_queries['query_id'] == query_id, text_column]
    if query_row.empty:
        raise ValueError(f"Query ID {query_id} not found in df_queries.")
    query = query_row.values[0]

    # Get top document indices from BM25
    bm25_scores, bm25_top_idx = BM25search.search(query, top_k)
    
    doc_candidates = [texts[idx] for idx in bm25_top_idx]
    candidate_idx = list(bm25_top_idx)
    
    top_n_scores, top_n_idx = colbert_reranker_batch.reranker(query, doc_candidates, candidate_idx, batch_size = 1, top_n=50)
    
    top_ranked_ids = []
    for idx in top_n_idx:
        idx = int(idx)
        top_ranked_ids.append(df_docs.iloc[idx]['doc_id'])

    for rank, doc_id in enumerate(top_ranked_ids):
        results.append({'query_id': query_id, 'doc_id': doc_id, 'rank': rank})

df_results = pd.DataFrame(results)

evaluation = IREvaluation()

k_set = [1, 5, 10, 20, 50]
for k in k_set:
    evaluation.compute_evaluation_metrics(df_qrels, df_results, k)