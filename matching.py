from sklearn.metrics.pairwise import cosine_similarity
from query_processing import process_query

def search(query, tfidf_matrix, vectorizer, documents):
    query_vector = process_query(query, vectorizer)
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = similarities.argsort()[::-1]
    return documents.iloc[ranked_indices], similarities[ranked_indices]


# استعلام مثال
# query = "example search query"
# results_data1, scores_data1 = search(query, tfidf_matrix_data1, vectorizer1, data1)
# results_data2, scores_data2 = search(query, tfidf_matrix_data2, vectorizer2, data2)

# print("Results from Data1:")
# print(results_data1.head())
# print("\nResults from Data2:")
# print(results_data2.head())