import os
from sklearn.feature_extraction import text
from nltk.corpus import stopwords

# An import that should function both locally and when running an a remote server
try:
    from environment_configuration import *
except:
    from topics2themes.environment_configuration import *

if RUN_LOCALLY:
    from topic_model_constants import *
    from word2vec_term_similarity import *

else:
    from topics2themes.topic_model_constants import *
    from topics2themes.word2vec_term_similarity import *
    

"""
Nr of topics to retrieve
"""
NUMBER_OF_TOPICS = 25


"""
The topic modelling algorithm is rerun with a decrease number of requested topics
until the number of found stable topics are similar to the ones requested
The amont of similarity is set here.
"""
#PROPORTION_OF_LESS_TOPIC_TO_ALLOW = 0.3
PROPORTION_OF_LESS_TOPIC_TO_ALLOW = 0.9

"""
Nr of words to display for each topic
"""
#NR_OF_TOP_WORDS = 10
NR_OF_TOP_WORDS = 10

"""
Nr of most typical document to retrieve for each topic
"""
#NR_OF_TOP_DOCUMENTS = 50
NR_OF_TOP_DOCUMENTS = 40

"""
Number of runs to check the stability of the retrieved topics.
Only topics that occur in all NUMBER_OF_RUNS runs will be
considered valid
"""
NUMBER_OF_RUNS = 50


"""
Mininimum overlap of retrieved terms to considered the retrieved topic as
the same topic of a another one
"""
OVERLAP_CUT_OFF = 0.70


"""
Whether to use pre-processing (collocation detection and synonym clustering)
"""
PRE_PROCESS = True

VECTOR_LENGTH = 100
SPACE_FOR_PATH = "/Users/marsk757/wordspaces/69/model.bin"
#MAX_DIST_FOR_CLUSTERING = 0.55
MAX_DIST_FOR_CLUSTERING = 0.60
WORDS_NOT_TO_INCLUDE_IN_CLUSTERING_FILE = "not_cluster.txt"
MANUAL_CLUSTER_FILE = "manual_clusters.txt"

"""
Mininimum occurrence in the corpus for a word to be included in the topic modelling
"""
MIN_DOCUMENT_FREQUENCY = 20

"""
Maximum occurrence in the corpus for a term to be included in the topic modelling
"""
MAX_DOCUMENT_FREQUENCY = 0.95

"""
Mininimum occurrence in the corpus for a term to be included in the clustering.
"""
MIN_DOCUMENT_FREQUENCY_TO_INCLUDE_IN_CLUSTERING = 10

"""
The stop word file of user-defined stopiwords to use (Scikit learn stop words are also used)
"""

STOP_WORD_FILE = "legends_stopwords.txt"



"""
The directories in which data is to be found. The data is to be in files with the ".txt" extension
in these directories. For each directory, there should also be a stance-label and a color associated with
the data
"""

DATA_LABEL_LIST = [{DATA_LABEL : "Hum", DIRECTORY_NAME : "hum", LABEL_COLOR : GREEN },\
                   {DATA_LABEL : "Neu", DIRECTORY_NAME : "neutral", LABEL_COLOR : "#ccad00"},\
                   {DATA_LABEL : "Med", DIRECTORY_NAME : "med", LABEL_COLOR : RED}]



TOPIC_MODEL_ALGORITHM = NMF_NAME
#TOPIC_MODEL_ALGORITHM = LDA_NAME


MAX_NR_OF_FEATURES = 10000

#STOP_WORD_SET = set(["many", "child", "kids", "parent", "types"])
#STOP_WORD_SET = set(stopwords.words('english'))

# Needed to do nltk.download('stopwords') to get it to work
STOP_WORD_SET = set(stopwords.words('swedish'))
print("STOP_WORD_SET", STOP_WORD_SET)


SHOW_ARGUMENTATION = False
SHOW_SENTIMENT = False


REMOVE_DUPLICATES = True

MIN_NGRAM_LENGTH_FOR_DUPLICATE = 15

BINARY_TF = False

def corpus_specific_text_cleaning(text):
    text = text.strip()
    text = "<p>" + text + "</p>"
    text = text.replace("\n\n", "\n")
    text = text.replace("\n", "</p><p>")
    return text
    
CLEANING_METHOD = corpus_specific_text_cleaning




