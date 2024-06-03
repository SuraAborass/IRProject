import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from num2words import num2words
import datefinder

nltk.download('stopwords')
nltk.download('wordnet')

# قاموس لتحويل أسماء الشهور النصية إلى أرقام
dic_month = {
    'january': '1', 'february': '2', 'march': '3', 'april': '4',
    'may': '5', 'june': '6', 'july': '7', 'august': '8',
    'september': '9', 'october': '10', 'november': '11', 'december': '12',
    'jan': '1', 'feb': '2', 'mar': '3', 'apr': '4', 'jun': '6', 'jul': '7',
    'aug': '8', 'sep': '9', 'oct': '10', 'nov': '11', 'dec': '12'
}
def date_processor(text):
    try:
        match = datefinder.find_dates(text, source=1)

        for item in match:
            replacement = ""

            if (item[1].__len__() > 4) and not ('PM' in item[1]) and not ('AM' in item[1]) and (
                    any(ch.isdigit() for ch in item[1])):
                replacement += str(item[0].day)
                replacement += "/"
                if type(item[0].month) == int:
                    replacement += str(item[0].month)
                else:
                    replacement += dic_month[item[0].month]
                replacement += "/"
                replacement += str(item[0].year)
                text = text.replace(item[1], replacement)

    except:
        pass

    return text



def convertNumToWords(text):
    for i in text.split():
        if i.isdigit():
            word = num2words(i)
            text = text.replace(i, word)
    return text

# تعريف دالة المعالجة
def preprocess_text(text):
    if isinstance(text, str):
        # تحويل النص إلى حروف صغيرة
        text = text.lower()
        text = text.replace('$', ' dollar ')
        text = text.replace('%', ' percent ')
        # إزالة علامات الترقيم
        text = text.translate(str.maketrans('', '', string.punctuation))
        # معالجة التواريخ
        text = date_processor(text)
        # تحويل الأرقام الى نصوص
        text = convertNumToWords(text)
        # تقطيع النص إلى كلمات
        tokens = nltk.word_tokenize(text)
        # إزالة الكلمات التوقف
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        # تطبيق الاستنباط (Stemming)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        # تطبيق الردمجة (Lemmatization)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    else:
        return ""

def preprocess_csv_file(output_path, num_rows):
    data1 ='C:/Users/Lenovo/OneDrive/سطح المكتب/PROJECT/antique/antique.csv'
    #data2 = pd.read_csv('wiklr/wikir.csv', nrows=num_rows)
    # معالجة النصوص في كل مجموعة بيانات
    #data1['processed_text'] = data1['text'].apply(preprocess_text)
    #data2['processed_text'] = data2['text'].apply(preprocess_text)

    # قراءة الملف بتنسيق CSV باستخدام مكتبة Pandas
    csv_table = pd.read_csv(data1,nrows=num_rows)
    print(csv_table.columns)
    # معالجة البيانات كما في الإصدار السابق
    csv_table['text'] = csv_table['text'].apply(preprocess_text)
    # حفظ البيانات المعالجة في ملف CSV جديد
    csv_table.to_csv(output_path, index=False)
