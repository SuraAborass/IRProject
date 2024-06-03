from preprocess_text import preprocess_text

# def process_query(query, vectorizer):
#     processed_query = preprocess_text(query)
#     print(processed_query)
#     return vectorizer.transform([processed_query])

def process_query(query, vectorizer):
    processed_query = preprocess_text(query)
    # print("Processed Query:", processed_query)  # طباعة الاستعلام المعالج
    query_vector = vectorizer.transform([processed_query])
    # print("Query Vector:", query_vector)  # طباعة النموذج المتجهي للاستعلام
    return query_vector
