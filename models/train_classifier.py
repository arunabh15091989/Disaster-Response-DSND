import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import TruncatedSVD
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,make_scorer

#sqlite:///..
nltk.download('stopwords')

def load_data(database_filepath='/data/DisasterResponse.db'):
    engine = create_engine(f'sqlite:///{database_filepath}')
    query =  'SELECT * FROM Table1'
    df = pd.read_sql(query,engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    stop_list = stopwords.words('english')
    #Normalize text
    norm_words = re.sub(r'[^a-zA-Z0–9]',' ',text)
    #Tokenze words
    words = word_tokenize(norm_words)
    #Stop words 
    words = [w for w in words if w not in stop_list]
    #Lemmatize
    lemmed = [WordNetLemmatizer().lemmatize(w, pos="v") for w in words]
    return lemmed


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # pipeline.fit(X_train,y_train)
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    metrics = []
    # Calculate evaluation metrics for each set of labels
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
        precision = precision_score(y_test[:, i], y_pred[:, i],average='weighted')
        recall = recall_score(y_test[:, i], y_pred[:, i],average='weighted')
        f1 = f1_score(y_test[:, i], y_pred[:, i],average='weighted')
        
        metrics.append([accuracy, precision, recall, f1])
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = category_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
    return metrics_df


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()