from src.logger import logging
from src.exception import Custom_Exception
import sys
import os
from sklearn.metrics import r2_score,f1_score
import pickle as pk
import numpy as np
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords



from sklearn.metrics import accuracy_score, f1_score

def evaluation(X_train, y_train, X_test, y_test, model):
    try:
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        report = {
            'model': model.__class__.__name__,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'f1_score_train': f1_score(y_train, y_pred_train, average='weighted'),
            'f1_score_test': f1_score(y_test, y_pred_test, average='weighted')
        }

        return report

    except Exception as e:
        raise Custom_Exception(e, sys)

    


def save_object(model_path, obj):
    try:
        path_name = os.path.dirname(model_path)
        os.makedirs(path_name, exist_ok=True)

        with open(model_path, 'wb') as file:
            pk.dump(obj, file)

    except Exception as e:
        raise Custom_Exception(e, sys)

    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pk.load(file_obj)
        
    except Exception as e:
        raise Custom_Exception(e, sys)

    



def avg_word2vec(words, model):
    """
    Calculates average Word2Vec vector for a list of tokens
    """

    if model is None:
        return np.zeros(100)  # Your vector_size - fallback
    
    valid_vectors = [model.wv[word] for word in words 
                     if word in model.wv.key_to_index]
    if not valid_vectors:
        return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)



def predict_pipeline_for_input(test_word):
        
        


    # Join the list into a single string separated by spaces
        single_line = " ".join(test_word)

        # (Optional) Remove existing internal newlines to ensure it is strictly one line
        clean_single_line = single_line.replace("\n", " ")

        word_in_text = clean_single_line.split()

        test_corpus = []
        wnl = WordNetLemmatizer()
        swords = list(stopwords.words('english'))
        swords.append('\n')




        for i in range(0,len(word_in_text)):
            review = re.sub('[^a-z A-Z 0-9]',' ',word_in_text[i])
            review = review.lower()
            review = review.split()
            review = [wnl.lemmatize(word_in_text) for word_in_text in review if word_in_text not in swords ]
            if len(review) == 0:
                i += 1
                continue

            test_corpus.append((review))
        return test_corpus
      
    
    



