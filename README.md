# Twitter Discourse and Emotions Around the Invasion of Ukraine
This repository contains all the code I wrote and used in my master thesis Twitter _Discourse and Emotions Around the Invasion of Ukraine – A Text Analytics Approach_. 

A companion website with interactive plots can be found [here](https://gabriellindelof.github.io/master_thesis_ukraine_invasion/)

Bellow is a summary of the contents of each code file: 

### scrape_main_dataset.bat
Batch script used to collect the main dataset from the Twitter API. Scrapes the same number of tweets for each hour of the selected day. 

### scrape_secondary_dataset.ipynb
Jupyter notebook used to collect the followees of users, as well as the tweets these followees made in the hour preceding the response tweet. 

### convert_main_dataset_csv.bat
Convert the scraped tweets to a more manageable CSV format. 

### data_processing.ipynb
Notebook that was used to clean the dataset and visualize the number of tweets before and after. 

### estimate_emotions_roberta.py
Python script that was run on a cloud computing cluster to estimate the emotions of tweets using Roberta. 

### estimate_primary_emotions.py
Python script that was run on a cloud computing cluster to count words associated with each of 8 primary emotions.

### fit_topic_model.py
Python script that was run on a cloud computing cluster to fit the BERTopic model on 1 million tweets. 

### transform_documents.py
This was used to assign all 8 million tweets with a topic from the model fitted in the script above. 

### sentiment_analysis.ipynb
A majority of the analysis made in this thesis is contained in this notebook. 
- Classification of valence and intensity
- Effect of intensity on retweets. 
- Comparisons of polarity, intensity, and specific emotions between topics. 
- Contagion of emotions using lexical-based as well as machine learning approach. 

### topic_model_analysis.ipynb
Contains visualizations and analysis of topic model. 

# Packages used: 
- Pandas
- Pyarrow
- Seaborn
- Numpy
- Matplotlib
- BERTopic
- HDBSCAN
- Twarc2
- Sci-kit learn
- Transformers
- sentence_transformers
- NLTK
- Datetime
- SciPy
- Statsmodels
- Pingouin
- WordCloud
- Gensim
- re
- csv
- docx
- Pickle
- Tqdm
- os
- json
