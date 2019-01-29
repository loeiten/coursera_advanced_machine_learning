import nltk
import pickle
import re
import numpy as np
import pandas as pd

nltk.download('stopwords')
from nltk.corpus import stopwords


# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': ['resource', 'intent_recognizer.pkl'],
    'TAG_CLASSIFIER': ['resource', 'tag_classifier.pkl'],
    'TFIDF_VECTORIZER': ['resource', 'tfidf_vectorizer.pkl'],
    'THREAD_EMBEDDINGS_FOLDER': ['resource', 'thread_embeddings_by_tags'],
    'WORD_EMBEDDINGS': ['resource', 'starspace_embedding.tsv'],
}


def text_prepare(text):
    """
    Performs tokenization and simple preprocessing.
    
    Parameters
    ----------
    text : str
        The text to prepare
        
    Returns
    -------
    str
        The prepared text
    """
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """
    Loads pre-trained word embeddings from tsv file.

    Parameters
    ----------
    embeddings_path : Path
        Path to the embeddings file.

    Returns
    -------
    embeddings : dict
        Mapping of words to vectors
    embeddings_dim : int 
        Dimension of the vectors.
    """
    
    embeddings_df = pd.read_csv(embeddings_path, 
                                sep='\t',
                                header=None)

    # Convert to dict
    words = embeddings_df.loc[:, 0].values
    vectors = embeddings_df.loc[:, 1:].values
    vectors = vectors.astype(np.float32)

    embeddings = dict()
    
    for word, vector in zip(words, vectors):
        embeddings[word] = vector

    # Extract dimension
    embeddings_dim = vectors.shape[1]
    return embeddings, embeddings_dim


def question_to_vec(question, embeddings, dim=300, verbose=False):
    """
    Transform a question to a string by taking the mean
    
    Parameters
    ----------
    question : str
        The quering question
    embeddings : dict-like
        A dict-like structure where the key is a word and its value is the
        embedding
    dim : int
        Size of the representation
    verbose : bool
        Whether or not to print information

    Returns
    -------
    result : np.array
        The vector representation of the question
    """

    result = np.zeros(dim)
    
    count = 0
    for word in question.split():
        count += 1
        try:
            result += embeddings[word]
        except KeyError:
            if count > 0:
                # Subtract expected count
                count -= 1
            if verbose:
                print('"{}" not in vocabulary'.format(word))
    
    if count > 0:
        result /= count
            
    return result

    
def pickle_file(content, path):
    """
    Pickle dumps a file
    
    Parameters
    ----------
    content : object
        Content to pickle
    path : Path
        Path to the file
    """
    
    with path.open('wb') as f:
        pickle.dump(content, f, pickle.HIGHEST_PROTOCOL)  

        
def unpickle_file(path):
    """
    Returns the result of unpickling the file content.
    
    Parameters
    ----------
    path : Path
        Path to the file
    
    Returns
    -------
    object
        The unpickled file
    """
    with path.open('rb') as f:
        return pickle.load(f)
