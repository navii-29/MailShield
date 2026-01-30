# basic imports
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import Custom_Exception
import sys
import os
import re
from src.utils import save_object,avg_word2vec



from gensim.models import Word2Vec
import nltk
import tqdm

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from dataclasses import dataclass

# merging with other py files

@dataclass
# model_path = '/Users/navie/Documents/Complete_MachineLearning_saved/Machine_Learning/NLP/archive-3/GoogleNews-vectors-negative300.bin'
class DataTransformationConfig:
    word_2_vec_obj_file_path: str = os.path.join("artifacts", "word2vec.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, raw_df):
        logging.info('Text preprocessing started')
        try:
            wnl = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))

            corpus = []



            for i in range(0,len(raw_df.text_combined)):
                review = re.sub('[^a-z A-Z 0-9]','',raw_df.text_combined[i])
                review = review.lower()
                review = review.split()
                review = [wnl.lemmatize(word) for word in review if word not in stop_words ]

                corpus.append((review))


            return corpus

        except Exception as e:
            raise Custom_Exception(e, sys)

    # --------------------------------------------------

    def vectorization_of_text(self, corpus):
        logging.info('Training Word2Vec model')
        try:
            
            model = Word2Vec(
                sentences=corpus,
                vector_size=100,
                window=5,
                min_count=2,
                workers=4,
                epochs=10
            )

            # training model from scratch



            save_object(model_path=self.data_transformation_config.word_2_vec_obj_file_path, obj = model)

            return model

        except Exception as e:
            raise Custom_Exception(e, sys)

    # --------------------------------------------------

    def vector_transformation(self, corpus, model, raw_df):
        logging.info('Creating Avg Word2Vec features')
        try:



            X = []
            for sentence in tqdm.tqdm(corpus):
                X.append(avg_word2vec(sentence, model))


            reshaped_data = [pd.DataFrame(item.reshape(1, -1)) for item in X]

            # Concatenate all DataFrames in the list into a single DataFrame
            df = pd.concat(reshaped_data, ignore_index=True)

            # X = np.array(X)

            target = (raw_df['label'] == 1).astype(int).values 
            y = pd.DataFrame(target) 
            X_df = pd.concat([df, y], axis=1)

            return X_df
           

        except Exception as e:
            raise Custom_Exception(e, sys)















