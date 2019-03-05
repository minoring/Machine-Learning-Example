import csv
import pickle
import gensim

from gensim import models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

dataset_file_name = './dataset/corpus.csv'
model_file_name = 'logistic.pk1'

# Change csv file format

with open(dataset_file_name, 'r') as f:
    csv.field_size_limit(1000000)
    document_sentence_data = list(csv.reader(f))

texts = []
label_ids = []
id_of_label = {}
IDX_OF_LABEL, IDX_OF_SENTENCES = 0, 1
sum_letters = []

for counter, row in enumerate(document_sentence_data):
    if counter == 0:
        continue
        
    label = row[IDX_OF_LABEL]
    
    if label not in id_of_label:
        # If Label dose not have id (e.g. 1, 2, ...) create id.
        id_of_label[label] = len(id_of_label)
    
    label_ids.append(id_of_label[label])
    word_list = row[IDX_OF_SENTENCES].split(' ')
    texts.append(word_list)
    
label_of_id = { ids: label for label, ids in id_of_label.items() }

# Split train data and test data.
X_train_texts, X_test_texts, y_train, y_test = \
        train_test_split(
            texts, 
            label_ids, 
            test_size=0.2,
            random_state=42)

# From training text data, create matrix which weighed TF-IDF.
text_data_dic = gensim.corpora.Dictionary(X_train_texts)

# Create corpus Bag of words of dictionary of texts.
corpus = [ text_data_dic.doc2bow(text) for text in X_train_texts ]

# Weight corpus using TF-IDF.
tfidf_model = models.TfidfModel(corpus)
tfidf_corpus = tfidf_model[corpus]

num_words = len(text_data_dic)
X_train_tfidf = gensim.matutils.corpus2dense(
        tfidf_corpus, 
        num_terms=num_words,
        ).T

# From text data, create matrix which weighed by TF-IDF

corpus = [ text_data_dic.doc2bow(text) for text in X_test_texts ]

# Weight corpus using TF-IDF.
tfidf_corpus = tfidf_model[corpus]

num_words = len(text_data_dic)
X_test_tfidf = gensim.matutils.corpus2dense(
        tfidf_corpus, 
        num_terms=num_words,
        ).T

clf = LogisticRegression(C=1)
clf.fit(X_train_tfidf, y_train)

# Evaluate classifier using test data.
y_pred = clf.predict(X_test_tfidf)
target_names = list(label_of_id.values())

print(classification_report(
    y_test,
    y_pred,
    target_names=target_names))

print(confusion_matrix(y_test, y_pred))