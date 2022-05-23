from sklearn.feature_extraction.text import CountVectorizer
from pyarrow import feather
import pandas as pd

def do_nothing(tokens):
    return tokens


lex = pd.read_csv('data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', sep='\t', names = ['word', 'emotion', 'present']) # Load emotion lexicon. 
lex = lex[lex.present == 1] # Only save rows where an associated emotion exists. 

# Create a dictonary containing emotions as keys and all associated words as values. 
em_dict = {}
for emotion in lex.emotion.unique():
    em_dict[emotion] = lex[(lex.emotion == emotion)]['word'].to_list()
    
tok = feather.read_feather('data/ukraine_two_weeks_clean_shuffled_v2_tok.feather') # Load tokenized tweets. 

for emotion in lex.emotion.unique(): # Iterate all 8 emotions. 
    print(emotion)
    # Count occurences of all words associated with specific emotion. (vocabulary limited to em_dict[emotion])
    vectorizer = CountVectorizer(vocabulary = em_dict[emotion], lowercase=False, preprocessor=lambda x: x, tokenizer=do_nothing) 
    vec = vectorizer.fit_transform(tok.tok)
    counts = [v.sum() for v in vec] # Sum the total number of words associated with emotion. 
    tok[emotion] = counts

print('Saving...')
feather.write_feather(tok, 'data/ukraine_two_weeks_clean_shuffled_v2_8emotions.feather')   
print('Done.')