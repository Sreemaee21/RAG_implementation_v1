from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import LdaModel
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel

# Download NLTK resources (run only once)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

with open('/mnt/c/llm_SERC_documentation/sree_llm_chatbot/RAG_assignment/aws_text.txt', 'r') as file:
    aws_text = file.read()

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove punctuation
    tokens = [word for word in tokens if word.isalnum()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


# Preprocess the AWS text
processed_aws_text = preprocess_text(aws_text)

# Create dictionary mapping of words to IDs for AWS text
dictionary = corpora.Dictionary([processed_aws_text])

# Create a document-term matrix for the AWS text
doc_term_matrix_aws = [dictionary.doc2bow(processed_aws_text)]

# Number of topics
num_topics = 50

# Build LDA model for "aws.txt"
lda_model_aws = LdaModel(doc_term_matrix_aws, num_topics=num_topics, id2word=dictionary, passes=20)

# Print the topics for "aws.txt"
for idx, topic in lda_model_aws.print_topics(-1):
    print(f'Topic {idx}: {topic}')
    
    
    
    
    
    






