from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from preprocess_text import preprocess_text
import pickle



# دمج النصوص من كلا مجموعتي البيانات لإنشاء تمثيل مشترك
# all_texts = pd.concat([data1['processed_text'], data2['processed_text']])
data1 = pd.read_csv('C:/Users/Lenovo/OneDrive/سطح المكتب/PROJECT/antique/antique_All_prosseced.csv')

#data1 ='C:/Users/Lenovo/OneDrive/سطح المكتب/PROJECT/antique/antique_All_prosseced.csv'

# إنشاء مصفوفة TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data1['text'])

# تقسيم المصفوفة إلى مجموعتين مرة أخرى
# tfidf_matrix_data1 = tfidf_matrix[:len(data1)]
# tfidf_matrix_data2 = tfidf_matrix[len(data1):]

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


with open('tfidf_vectorizer_data1.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
# with open('tfidf_vectorizer_data2.pkl', 'wb') as f:
#     pickle.dump(vectorizer2, f)
with open('tfidf_matrix_data1.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)
# with open('tfidf_matrix_data2.pkl', 'wb') as f:
#     pickle.dump(tfidf_matrix_data2, f)

# استعلام مثال
# query = "example search query"
# results_data1, scores_data1 = search_query(query, tfidf_matrix, data1)
# # results_data2, scores_data2 = search_query(query, tfidf_matrix_data2, data2)

# # عرض النتائج
# print("Results from Data1:")
# print(results_data1.head())
# print("\nResults from Data2:")
# print(results_data2.head())