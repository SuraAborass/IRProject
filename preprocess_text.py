import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from num2words import num2words
import datefinder
from decimal import Decimal, InvalidOperation

nltk.download('stopwords')
nltk.download('wordnet')

# Dictionary to convert month names to numbers
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
            if (item[1].len() > 4) and not ('PM' in item[1]) and not ('AM' in item[1]) and (
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
            try:
                number = int(i)
                if abs(number) < 10**15:  # Handle numbers smaller than 10^15
                    word = num2words(number)
                    text = text.replace(i, word)
            except (InvalidOperation, OverflowError):
                continue
    return text

# Define the preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert text to lowercase
        text = text.lower()
        text = text.replace('$', ' dollar ')
        text = text.replace('%', ' percent ')
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Process dates
        text = date_processor(text)
        # Convert numbers to words
        #text = convertNumToWords(text)
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        # Apply stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        # Apply lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    else:
        return ""

def preprocess_csv_file(input_path, output_path, num_rows):
    # Read the CSV file using Pandas
    data1 = pd.read_csv(input_path, nrows=num_rows)
    # Process the text in the DataFrame
    data1['processed_text'] = data1['text'].apply(preprocess_text)
    # Save the processed data to a new CSV file
    data1.to_csv(output_path, index=False)
#preprocess_csv_file('C:/Users/Lenovo/OneDrive/سطح المكتب/PROJECT/antique/antique.csv', 'C:/Users/Lenovo/OneDrive/سطح المكتب/PROJECT/antique/antique_All_prosseced.csv', 10)