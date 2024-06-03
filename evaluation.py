import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

def load_relevance(filepath):
    # Load the relevance data
    df = pd.read_csv(filepath)
    return df

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
        relevant_docs = relevance_data[relevance_data['query_id'] == query_id]

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
        
    # Format the scores to 5 decimal places
    evaluation_scores = {k: format(v, '.5f') for k, v in evaluation_scores.items()}
    map_scores = [format(score, '.5f') for score in map_scores]
    mrr_scores = [format(score, '.5f') for score in mrr_scores]
    recall_scores = [format(score, '.5f') for score in recall_scores]
    precision_scores = [format(score, '.5f') for score in precision_scores]
    
    return evaluation_scores, map_scores, mrr_scores, recall_scores, precision_scores

# Example usage
# query_similarities = process_query()
# relevance_data = load_relevance('C:/Users/Lenovo/OneDrive/سطح المكتب/PROJECT/antique/train/relevance.csv')
# evaluation_scores, map_scores, mrr_scores, recall_scores, precision_scores = calculate_evaluation_scores(query_similarities, relevance_data)
# print(evaluation_scores)
# print(map_scores)
# print(mrr_scores)
# print(recall_scores)
# print(precision_scores)