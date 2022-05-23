print("Starting imports...")
from pyarrow import feather
from bertopic import BERTopic
import os
import pickle
import pandas as pd
from pyarrow import feather
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from hdbscan import HDBSCAN


print("Parallelism off...")
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Turn of since it causes issues with the cloud computing platform

modelname = 'modelname'

print("Initiating...")

# A shuffled version of the dataset is used so that we can pick the first 1000000, keeping track of which random documents model was trained on. 
df = feather.read_feather('data/ukraine_two_weeks_clean_shuffled_v2.feather', columns = ['id', 'text_clean']) 
df = df[:1000000]

docs = df.text_clean.tolist() # Get text content of documents
print("No. docs: ", len(docs))

print("DONE READING FILE")



print("FITTING MODEL...")
stop_words = stopwords.words('english') # add NLTK stop words
stop_words.append('ukraine') # add ukraine as stop-word, we don't want it since it is in every document 
vectorizer_model = CountVectorizer(stop_words = stop_words, min_df=0.00001) # create vectorizer model used with stop words and min-wird-frequency


#hdbscan_model = HDBSCAN(min_cluster_size=3000, min_samples=500) # used to modify HBDSCAN, not used in final model

topic_model = BERTopic(verbose=True, calculate_probabilities = True, min_topic_size = 3000, low_memory = True, vectorizer_model = vectorizer_model) # Initiate model



# load saved embeddings to save time (created first run)
embeddings = np.load('embeddings_1mil_v2.npy')

# fit the model
topics, probs = topic_model.fit_transform(docs, embeddings = embeddings)


print("SAVING MODEL...")
topic_model.save('models/{}'.format(modelname), save_embedding_model=False)
print("DONE SAVING.")


print("SAVING TOPICS...")
filename = 'models/{}.pickle'.format(modelname)
with open(filename, 'wb') as f:
    pickle.dump(topics, f)
print("DONE TOPICS.")

 
    
print("SAVING PROBS...")
filename = 'models/{}_probs.pickle'.format(modelname)
with open(filename, 'wb') as f:
    pickle.dump(probs, f) 
print("DONE SAVING.")

