import nltk
import scipy
from scipy.sparse.linalg import svds

import preprocessing

"""
nltk library is required, to download it use this command:
pip install --user -U nltk 
"""


def get_text_vector(dictionary, text):
    """ Creating a bag-of-words for text to search it. """

    # stop words to remove them from the text to search
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # object which will stem the words
    porter_stemmer = nltk.stem.porter.PorterStemmer()

    # text split on words using nltk library
    word_tokens = nltk.tokenize.word_tokenize(text)

    # filtering words to remove these which are "stop words"
    filter_words = list(filter(lambda w: w not in stop_words, word_tokens))

    # stemming words
    stemmed_words = list(map(lambda w: porter_stemmer.stem(w), filter_words))

    # bag-of-words vector of the text to search
    text_vector = scipy.sparse.lil_matrix((len(dictionary), 1), dtype=float)

    # adding words occurrence to the vector
    for word in stemmed_words:
        if word in dictionary:
            text_vector[dictionary[word], 0] += 1

    # converting to csr matrix for faster operations
    text_vector = scipy.sparse.csc_matrix(text_vector)

    # normalization
    text_vector /= scipy.sparse.linalg.norm(text_vector)

    return text_vector


def search(dictionary, matrix, text, k, return_res=False):
    """ Searching word and returning k texts with the highest probability to contain that text. """

    # getting bag-of-words vector for the text to search
    text_vector = get_text_vector(dictionary, text)

    # getting correlation for every article
    correlation = []
    for i in range(matrix.shape[1]):
        word_correlation = (text_vector[:, 0].T @ matrix[:, i])[0, 0]

        correlation.append((word_correlation, i))

    # sorting to get the most suitable articles
    correlation.sort(reverse=True)

    if return_res:
        # returning result
        return list(map(lambda item: f"Article: texts/{item[1]}.txt, correlation: {item[0]}", correlation[:k]))
    else:
        # printing result
        for i in range(k):
            print(f"Article: texts/{correlation[i][1]}.txt, correlation: {correlation[i][0]}")


def preprocess():
    return preprocessing.get_matrix_and_dictionary(use_idf=True, reduce_noise=True, k=250)


def main_func():
    matrix, dictionary = preprocessing.get_matrix_and_dictionary(use_idf=True, reduce_noise=True, k=250)

    text = "is of the first"
    print(search(dictionary, matrix, text, 3, True))
