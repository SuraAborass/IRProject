from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score, precision_score, recall_score
import pandas as pd
from preprocess_text import preprocess_text
import pickle
import csv
import numpy as np

data1 = pd.read_csv('C:/Users/Lenovo/OneDrive/سطح المكتب/PROJECT/antique/antique_All.csv')
data1 = data1.dropna(subset=['text'])

# إنشاء مصفوفة TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data1['text'])

def search_query(query, tfidf_matrix, documents):
    # معالجة الاستعلام
    processed_query = preprocess_text(query)
    # تحويل الاستعلام إلى تمثيل TF-IDF
    query_vector = vectorizer.transform([processed_query])
    # حساب التشابه بين الاستعلام والوثائق
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    # ترتيب الوثائق بناءً على التشابه
    ranked_indices = similarities.argsort()[::-1]
    return documents.iloc[ranked_indices], similarities[ranked_indices]

def load_queries(filepath):
    queries = []
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            queries.append(row)
    return queries

def calculate_evaluation_scores(query_similarities, relevance_data):
    evaluation_scores = {
        "MAP": 0,
        "MRR": 0,
        "Recall": 0,
        "Precision": 0,
    }

    map_scores = []
    mrr_scores = []
    recall_scores = []
    precision_scores = []

    for query_id, similarities in query_similarities.items():
        # Get the relevant documents for this query
        relevant_docs = relevance_data[relevance_data['query_id'] == int(query_id)]

        if relevant_docs.empty:
            continue

        # Sort the similarities and get the ranking of document IDs
        sorted_indices = np.argsort(-similarities)  # Descending order
        ranked_doc_ids = sorted_indices + 1  # Assuming document IDs are 1-indexed

        # Get the relevance of the ranked documents
        relevance = [1 if doc_id in relevant_docs['doc_id'].values else 0 for doc_id in ranked_doc_ids]

        # Calculate Average Precision (AP)
        if any(relevance):
            ap = average_precision_score(relevance, similarities[sorted_indices])
            map_scores.append(ap)

        # Calculate Mean Reciprocal Rank (MRR)
        try:
            first_relevant_rank = next(i for i, r in enumerate(relevance, 1) if r)
            mrr_scores.append(1 / first_relevant_rank)
        except StopIteration:
            mrr_scores.append(0)

        # Calculate Precision and Recall for the top-k (e.g., top-10)
        k = 10
        top_k_relevance = relevance[:k]
        precision = precision_score(relevant_docs['relevance'], top_k_relevance, zero_division=0)
        recall = recall_score(relevant_docs['relevance'], top_k_relevance, zero_division=0)
        
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Calculate the mean of each metric, avoiding empty slices
    if map_scores:
        evaluation_scores['MAP'] = np.mean(map_scores)
    if mrr_scores:
        evaluation_scores['MRR'] = np.mean(mrr_scores)
    if recall_scores:
        evaluation_scores['Recall'] = np.mean(recall_scores)
    if precision_scores:
        evaluation_scores['Precision'] = np.mean(precision_scores)
        
    return evaluation_scores, map_scores, mrr_scores, recall_scores, precision_scores

def process_query():
    # Load the queries
    queries1 = load_queries('C:/Users/Lenovo/OneDrive/سطح المكتب/PROJECT/antique/train/queries.csv')
    queries2 = load_queries('C:/Users/Lenovo/OneDrive/سطح المكتب/PROJECT/wiklr/training/queries.csv')
    # Load the vectorizer and tfidf matrix
    with open('tfidf_vectorizer_data1.pkl', 'rb') as f:
        vectorizer1 = pickle.load(f)
    with open('tfidf_matrix_data1.pkl', 'rb') as f:
        tfidf_matrix_data1 = pickle.load(f)
    with open('tfidf_vectorizer_data2.pkl', 'rb') as f:
        vectorizer2 = pickle.load(f)
    with open('tfidf_matrix_data2.pkl', 'rb') as f:
        tfidf_matrix_data2 = pickle.load(f)

    results = {}
    all_evaluation_scores = []

    # Process queries from dataset 1
    for query in queries1:
        query_id = query['query_id']
        query_text = query['query_text']
        processed_query = preprocess_text(query_text)
        query_tfidf_vector = vectorizer1.transform([processed_query])
        query_cosin_similarities = cosine_similarity(query_tfidf_vector, tfidf_matrix_data1).flatten()
        
        # Get top 10 results
        top_10_indices = query_cosin_similarities.argsort()[-10:][::-1]
        top_10_similarities = query_cosin_similarities[top_10_indices]
        results[query_id] = top_10_similarities.tolist()

        # Calculate evaluation scores
        relevance_data = pd.read_csv('C:/Users/Lenovo/OneDrive/سطح المكتب/PROJECT/antique/train/qrels2.csv')
        evaluation_scores, _, _, _, _ = calculate_evaluation_scores({query_id: query_cosin_similarities}, relevance_data)
        all_evaluation_scores.append(evaluation_scores)
    
    # Process queries from dataset 2
    # for query in queries2:
    #     query_id = query['query_id']
    #     query_text = query['query_text']
    #     processed_query = preprocess_text(query_text)
    #     query_tfidf_vector = vectorizer2.transform([processed_query])
    #     query_cosin_similarities = cosine_similarity(query_tfidf_vector, tfidf_matrix_data2).flatten()
        
    #     # Get top 10 results
    #     top_10_indices = query_cosin_similarities.argsort()[-10:][::-1]
    #     top_10_similarities = query_cosin_similarities[top_10_indices]
    #     results[query_id] = top_10_similarities.tolist()
        
    #     # Calculate evaluation scores
    #     relevance_data = pd.read_csv('C:/Users/Lenovo/OneDrive/سطح المكتب/PROJECT/wiklr/training/relevance.csv')
    #     evaluation_scores, _, _, _, _ = calculate_evaluation_scores({query_id: query_cosin_similarities}, relevance_data)
    #     all_evaluation_scores.append(evaluation_scores)

    # Calculate mean of all evaluation scores
    mean_evaluation_scores = {
        "MAP": np.mean([score['MAP'] for score in all_evaluation_scores]),
        "MRR": np.mean([score['MRR'] for score in all_evaluation_scores]),
        "Recall": np.mean([score['Recall'] for score in all_evaluation_scores]),
        "Precision": np.mean([score['Precision'] for score in all_evaluation_scores])
    }
        
    return mean_evaluation_scores, all_evaluation_scores

# Example usage
# mean_evaluation_scores, all_evaluation_scores = process_query()
# print("Mean Evaluation Scores:", mean_evaluation_scores)
# print("All Evaluation Scores:")
# for scores in all_evaluation_scores:
#     print(scores)