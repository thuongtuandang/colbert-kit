from test_cb_tdqq.retrieving.colbert_search_faiss import colBERTSearchFaiss
from test_cb_tdqq.indexing.faiss.flat_faiss_index import FlatFaissIndex
from test_cb_tdqq.embedding.colbert_embedding import colBERTEmbedding
from test_cb_tdqq.indexing.bm25.bm_25 import BM25Corpus
from test_cb_tdqq.retrieving.bm25_search import BM25_search
from test_cb_tdqq.evaluating.ir_eval import IREvaluation
import pandas as pd
from tqdm import tqdm

eval_name = 'antique_de'
print("Test dataset: ", eval_name)
eval_data_path = '../data_test/' + f'{eval_name}/'
query_path = eval_data_path + 'queries.csv'
doc_path = eval_data_path + 'docs.csv'
qrel_path = eval_data_path + 'qrels.csv'

text_column = 'translated_text'

df_queries = pd.read_csv(query_path)
df_qrels = pd.read_csv(qrel_path)[:10]
df_docs = pd.read_csv(doc_path)[:30]

df_docs['index'] = df_docs.index
doc_ids = df_docs.set_index('index')['doc_id'].to_dict()
texts = df_docs[text_column].tolist()

# Colbert embedding and indexing
token_to_doc_map_name = 'token_to_doc_map.npy'
doc_dict_name = 'documents.json' 
token_to_doc_map_path = '../indices_test/'
doc_dict_path = '../indices_test/'
colbert_model_path = '../model/colbert'

colbertEmbedIndex = colBERTEmbedding(
    model_name_or_path=colbert_model_path,
    token_to_doc_map_name=token_to_doc_map_name,
    doc_dict_name=doc_dict_name,
    token_to_doc_map_path=token_to_doc_map_path,
    doc_dict_path=doc_dict_path,
    device="cpu"
)

embeddings = colbertEmbedIndex.encode(texts)

# Now, we need to create FAISS index to search for nearest tokens
index_output_path = "../indices_test/"
dimension = embeddings.shape[1]
index_name = 'colbert_antique_de.index'
# Create a flat index
colbert_index = FlatFaissIndex(index_name=index_name, dimension=dimension, use_gpu=False, index_output_path=index_output_path)
colbert_index.create_index(embeddings)


# Load colbert model for embedding
colbert_model_path = '../model/colbert'

colbertSearchFaiss = colBERTSearchFaiss(
    model_name_or_path=colbert_model_path,
    index_path=index_output_path,
    index_name=index_name,
    index_type='flat',
    use_gpu=False,
    token_to_doc_map_name=token_to_doc_map_name,
    doc_dict_name=doc_dict_name,
    token_to_doc_map_path=token_to_doc_map_path,
    doc_dict_path=doc_dict_path,
    device="cpu"
)

colbertSearchFaiss.load_index_map_and_dict()

# Compute metrics for BM25 search
print("Metrics for BM25 search")
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
top_k = 50

for query_id in tqdm(query_ids, desc="Processing queries"):
    query_row = df_queries.loc[df_queries['query_id'] == query_id, text_column]
    if query_row.empty:
        raise ValueError(f"Query ID {query_id} not found in df_queries.")
    query = query_row.values[0]

    # Get top document indices from BM25
    bm25_scores, bm25_top_idx = BM25search.search(query, top_k)
    
    top_ranked_ids = []
    for idx in bm25_top_idx:
        idx = int(idx)
        top_ranked_ids.append(df_docs.iloc[idx]['doc_id'])

    for rank, doc_id in enumerate(top_ranked_ids):
        results.append({'query_id': query_id, 'doc_id': doc_id, 'rank': rank})

df_results = pd.DataFrame(results)

evaluation = IREvaluation()

k_set = [1, 5, 10, 20, 50]
for k in k_set:
    evaluation.compute_evaluation_metrics(df_qrels, df_results, k)

print("Metrics for ColBERT search")
results = []
query_ids = df_qrels['query_id'].unique()
top_k = 50

# ColBERT retrieving metrics

for query_id in tqdm(query_ids, desc="Processing queries"):
    query_row = df_queries.loc[df_queries['query_id'] == query_id, text_column]
    if query_row.empty:
        raise ValueError(f"Query ID {query_id} not found in df_queries.")
    query = query_row.values[0]

    # Semantic search with colBERT
    colbert_top_idx, scores = colbertSearchFaiss.colbert_search_results(query,
                                                                    top_k_results=top_k,
                                                                    top_k_search_tokens=10)

    top_ranked_ids = []
    for idx in colbert_top_idx:
        idx = int(idx)
        top_ranked_ids.append(df_docs.iloc[idx]['doc_id'])
    
    for rank, doc_id in enumerate(top_ranked_ids):
        results.append({'query_id': query_id, 'doc_id': doc_id, 'rank': rank})

df_results = pd.DataFrame(results)

evaluation = IREvaluation()

k_set = [1, 5, 10, 20, 50]
for k in k_set:
    evaluation.compute_evaluation_metrics(df_qrels, df_results, k)