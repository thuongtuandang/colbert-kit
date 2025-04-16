from tqcolbert_src.retrieving.colbert_search_faiss import colBERTSearchFaiss
from tqcolbert_src.indexing.faiss.flat_faiss_index import FlatFaissIndex
from tqcolbert_src.indexing.faiss.HNSW_faiss_index import HNSWFaissIndex
from tqcolbert_src.indexing.faiss.IVF_faiss_index import IVFFaissIndex
from tqcolbert_src.embedding.colbert_embedding import colBERTEmbedding
import pandas as pd

# Load the csv file
data_path = '../data_test/miracl-de-dev/docs.csv'
df = pd.read_csv(data_path)
df = df.dropna()
df['index'] = df.index
# Create a text list from overview column
# This will be used for retrieval
texts = df.iloc[:30]['text'].tolist()

"""
Create index with colBERT

Note that indexing with colBERT is not true index, like FAISS or chromadb
ColBERT will compute score based on token semantic similarity
The base model is BERT
Note that BERT is semantic embedding, it means the embedding of I in the two sentences
"I am sad" and "I am happy" are different.

First retrieval with colBERT will have 3 phases:
Phase 1: for each token in the query, search for nearest tokens
Phase 2: map tokens back to their original documents
Phase 3: compute maxsim score between query and documents

So, we need to have token_to_doc_map and doc_dict 
doc_dict is a place where we save index of the form {'index': text}
"""

token_to_doc_map_name = 'token_to_doc_map.npy'
doc_dict_name = 'documents.json' 
token_to_doc_map_path = '../indices_test/'
doc_dict_path = '../indices_test/'
embedding_model_path = '../model/colbert'
# embedding_model_path = "google-bert/bert-base-german-cased"

colbertEmbed = colBERTEmbedding(
    model_name_or_path=embedding_model_path,
    token_to_doc_map_name=token_to_doc_map_name,
    doc_dict_name=doc_dict_name,
    token_to_doc_map_path=token_to_doc_map_path,
    doc_dict_path=doc_dict_path,
    device="cpu"
)

embeddings = colbertEmbed.encode(texts)

# Now, we need to create FAISS index to search for nearest tokens
index_output_path = '../indices_test/'
index_name = 'colbert_test_index.index'
dimension = embeddings.shape[1]
# Create an HNSW index
hnsw_index = HNSWFaissIndex(index_name=index_name, dimension=dimension, use_gpu=False, index_output_path=index_output_path)
hnsw_index.create_index(embeddings)

"""
We now go to the search phase with FAISS
Parameters:
model_name_or_path: colbert model's path
index_path, index_name, index_type, use_gpu: parameters for FAISS search
token_to_doc_map name and path: place where we save the token_to_doc_map
doc_dict_name and path: please where we save the document dictionary
"""

colbertSearchFaiss = colBERTSearchFaiss(
    model_name_or_path=embedding_model_path,
    index_path=index_output_path,
    index_name=index_name,
    index_type='HNSW',
    use_gpu=False,
    token_to_doc_map_name=token_to_doc_map_name,
    doc_dict_name=doc_dict_name,
    token_to_doc_map_path=token_to_doc_map_path,
    doc_dict_path=doc_dict_path,
    device="cpu"
)

colbertSearchFaiss.load_index_map_and_dict()

query_df = pd.read_csv('../data_test/miracl-de-dev/queries.csv')
for i, row in query_df.iloc[2:4].iterrows():
    query = row['text']
    print(f'Query: {query}')
    print("Searching...")
    scores, colbert_indices  = colbertSearchFaiss.colbert_search_results(query,
                                                                        top_k_results=5,
                                                                        top_k_search_tokens=10)
    print("Search complete!")
    results = []
    for i, result in enumerate(colbert_indices):
        idx = int(colbert_indices[i])
        metadata_info = {
            "doc": df.loc[idx, 'text'],
            'score': scores[i]
        }
        results.append(metadata_info)

    # Print the list of results with distances and metadata
    for result in results:
        print(result)
    print('--------------')