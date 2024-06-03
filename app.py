from flask import Flask, request, render_template
import pickle
import pandas as pd
from matching import search
# from query_processing import process_query
# from evaluation import calculate_map_mrr_recall_precision_at_k

app = Flask(__name__)
# تحميل الفهارس
with open('tfidf_vectorizer_data1.pkl', 'rb') as f:
    vectorizer1 = pickle.load(f)
with open('tfidf_vectorizer_data2.pkl', 'rb') as f:
    vectorizer2 = pickle.load(f)
with open('tfidf_matrix_data1.pkl', 'rb') as f:
    tfidf_matrix_data1 = pickle.load(f)
with open('tfidf_matrix_data2.pkl', 'rb') as f:
    tfidf_matrix_data2 = pickle.load(f)

data1 = pd.read_csv('C:/Users/Lenovo/OneDrive/سطح المكتب/PROJECT/antique/antique_All.csv')
data2 = pd.read_csv('C:/Users/Lenovo/OneDrive/سطح المكتب/PROJECT/wiklr/wikir_All_All.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_request():
    query = request.form['query']
    dataset = request.form['dataset']
    
    if dataset == 'data1':
        results, scores = search(query, tfidf_matrix_data1, vectorizer1, data1)
    elif dataset == 'data2':
        results, scores = search(query, tfidf_matrix_data2, vectorizer2, data2)

        # results, scores = search(query, tfidf_matrix_data2, vectorizer2, data2)
    
    # relevant_docs = [74, 209, 336]  # قائمة الوثائق ذات الصلة للاستعلام
    # evaluation_scores = calculate_map_mrr_recall_precision_at_k(results.index.tolist(), relevant_docs)
    
    return render_template('results.html', query=query, results=results.head(30).to_html())

if __name__ == '__main__':
    app.run(debug=True)