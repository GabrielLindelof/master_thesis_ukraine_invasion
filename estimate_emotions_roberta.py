from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
from pyarrow import feather
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic.backend._utils import select_backend
from bertopic import BERTopic
import pickle
import re
from tqdm import tqdm
tqdm.pandas()

def get_emotions(text):
    '''Encode text and predict emotions.'''
    text = re.sub('\\s+', ' ', text) # Remove redundant blank space.
    
    encoded_input = tokenizer(text, return_tensors='pt') # Encode text. 
    output = model(**encoded_input) # Get model prediction. 
    scores = output[0][0].detach().numpy()
    scores = softmax(scores) 
    
    # Return scores as pd.Series. 
    s = pd.Series(dtype = 'float64')
    s['anger'] = scores[0]
    s['joy'] = scores[1]
    s['optimism'] = scores[2]
    s['sadness'] = scores[3]
    return s



task='emotion'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}" # Set model name to download. 

tokenizer = AutoTokenizer.from_pretrained(MODEL) # Load tokenizer. 

# Download label mappings.
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt" 
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]


model = AutoModelForSequenceClassification.from_pretrained(MODEL) # Load model
model.save_pretrained(MODEL) # Save model locally. 





model_name = 'model_1mil_v13_14t' 
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic.load('models/{}'.format(model_name), embedding_model=sentence_model) # Load topic model.
topics = pickle.load(open('models/{}_all.pickle'.format(model_name),'rb')) # Load topic labels for tweets. 




columns = ['author_id', 'text', 'text_clean', 'public_metrics.retweet_count', 'public_metrics.like_count', 'author.public_metrics.followers_count']
df = feather.read_feather('data/ukraine_two_weeks_clean_shuffled_v2.feather', columns = columns) # Load tweets. 

# Add topic names to tweets. 
df['topic'] = topics 
df['topic_name'] = df.topic.apply(lambda x: topic_model.get_topic(x)[0][0])



# Sample 5000 tweets from each topic. 
topic_sample = df[['text_clean', 'topic', 'topic_name']].groupby('topic').sample(5000)

print("loaded data, starting classification...")
x = topic_sample.text_clean.progress_apply(get_emotions) # Run Roberta prediction on each tweet. 
topic_sample = topic_sample.join(x)

feather.write_feather(topic_sample, 'data/sample_topic_emotions_5000_each_14t.feather') # Save predictions. 

print("saving complete...")