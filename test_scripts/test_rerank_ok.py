from test_cb_tdqq.reranking.colbert_batch_reranker import colBERTReRankerBatch
 
embedding_model_path = "../model/colbert"
# embedding_model_path = "google-bert/bert-base-german-cased"
batchreranker = colBERTReRankerBatch(model_name_or_path=embedding_model_path, device='cpu')
 
query = "Was ist die Hauptstadt von Frankreich?"
 
doc_candidates = [
    "Paris ist die Hauptstadt von Frankreich.",
    "Berlin ist die Hauptstadt von Deutschland.",
    "Madrid ist die Hauptstadt von Spanien.",
    "Rom ist die Hauptstadt von Italien.",
    "Der Eiffelturm befindet sich in Paris.",
    "Deutschland ist ein Land in Europa mit Berlin als Hauptstadt.",
    "Spanien ist bekannt für seine lebendige Kultur und Madrid ist seine Hauptstadt.",
    "Italien hat eine reiche Geschichte und Rom ist seine Hauptstadt.",
    "Das Louvre-Museum befindet sich in Paris, Frankreich.",
    "Berlin hat viele historische Sehenswürdigkeiten.",
    "Madrid beherbergt den Königspalast.",
    "Rom hat das Kolosseum, eine ikonische historische Stätte.",
    "Paris ist berühmt für seine Kunst, Mode und Kultur.",
    "Berlin ist ein Zentrum für Technologie und Innovation in Europa.",
    "Madrid beherbergt das Prado-Museum, eines der besten Kunstmuseen der Welt.",
    "Rom beherbergt die Vatikanstadt innerhalb seiner Grenzen.",
    "Frankreich ist bekannt für seine Weine, wobei Paris die Hauptstadt ist.",
    "Deutschland ist berühmt für sein Bier und das Brandenburger Tor in Berlin.",
    "Spaniens Hauptstadt Madrid ist bekannt für ihr lebendiges Nachtleben.",
    "Italiens Hauptstadt Rom ist berühmt für ihre Renaissancekunst und Architektur.",
    "Die Kathedrale Notre-Dame ist ein berühmtes Wahrzeichen in Paris.",
    "Das Reichstagsgebäude in Berlin ist eine historische Stätte in Deutschland.",
    "Die Plaza Mayor in Madrid ist ein zentraler Platz von historischer Bedeutung.",
    "Der Trevi-Brunnen in Rom ist ein Muss für Touristen.",
]
candidate_idx = list(range(len(doc_candidates)))
 
top_scores, top_indices = batchreranker.reranker(query, doc_candidates, candidate_idx, batch_size=8, top_n=10)
print("Top Scores:", top_scores)
print("Top Indices:", top_indices)
for id in top_indices:
    print(doc_candidates[id])
