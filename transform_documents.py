from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic.backend._utils import select_backend
from pyarrow import feather
import pickle
import os

print("Parallelism off...")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

modelname = 'modelname'
batch = 'N' # # processing done in batches, 1 million at a time. This variable is used to save the batch number to the filename. 


print("Initiating...")
df = feather.read_feather('data/ukraine_two_weeks_clean_shuffled_v2.feather', columns = ['id', 'text_clean']) # load documents
df = df[1000000:2000000] # processing done in batches, 1 million at a time. Here the documents contained in this run is set. 

docs = df.text_clean.tolist() # clean documents

print("Nr docs to be transformed: ", len(docs))
sentence_model = SentenceTransformer("all-MiniLM-L6-v2") # load sentence transformer explicitly, since it is not saved with model.
model = select_backend(sentence_model) # Select it to the model

topic_model = BERTopic.load("models/{}".format(modelname), embedding_model=model) # load topic model, used to predict topics of documents.
topics, probs = topic_model.transform(docs) # Predict topic of all documents in batch N


print("SAVING TOPICS...")
filename = "models/{}_batch_{}.pickle".format(modelname, batch)
with open(filename, 'wb') as f:
    pickle.dump(topics, f)
print("COMPLETE")

print("SAVING PROBS...")
filename = "models/{}_batch_{}_probs.pickle".format(modelname, batch)
with open(filename, 'wb') as f:
    pickle.dump(probs, f)
print("COMPLETE")
