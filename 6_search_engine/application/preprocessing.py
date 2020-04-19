import nltk
import numpy as np
import scipy
from scipy.sparse.linalg import svds
import math

articles_num = 2000


def separate_text_file():
    """ Separating text.txt file with many articles into separate articles"""
    with open("text.txt") as file:
        text = file.read()

        # every article is starting from "@@[number]" so it is good to split text using "@@"
        articles = text.split("@@")

        # to remove initial number in every article
        numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        curr_article = 0

        for inx in range(1, len(articles)):
            if len(articles[inx]) > 972:    # to get exactly 2000 shortest articles from above 4000 in "text.txt"
                continue

            article_file = open("texts/" + str(curr_article) + ".txt", "w")

            # sometimes there is " @ @ @ @ " in text so here is removed
            articles[inx] = articles[inx].replace("@ ", "")

            # every article starts with number, this loop is to remove this number
            letter = 0
            while articles[inx][letter] in numbers:
                letter += 1

            # writing article to the file
            article_file.write(articles[inx][letter+1:])
            article_file.close()

            curr_article += 1


def make_dictionary_and_matrix():
    """ Making dictionary with all words in articles files in "texts" directory.
        Key is a word and value is its index.

        Then creating sparse term-by-document matrix.
        Matrix's row represents a word and column represents an article.
        matrix[i, j] = k means that there is k words having index i in dictionary in text "texts/j.txt"
    """

    # result dictionary with all words in all articles
    dictionary = {}

    # to have an index of next word in dictionary
    curr_word = 0

    # stop words to remove them from dictionary
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # object which will stem the words
    porter_stemmer = nltk.stem.porter.PorterStemmer()

    words_by_article = []
    for i in range(articles_num):
        with open("texts/" + str(i) + ".txt") as file:
            # article text from file
            text = file.read()

            # text split on words using nltk library
            word_tokens = nltk.tokenize.word_tokenize(text)

            # filtering words to remove these which are "stop words"
            filter_words = list(filter(lambda w: w not in stop_words, word_tokens))

            # stemming words
            stemmed_words = list(map(lambda w: porter_stemmer.stem(w), filter_words))

            # writing new words to dictionary
            for word in stemmed_words:
                if word not in dictionary:
                    dictionary[word] = curr_word
                    curr_word += 1

            # to write them later to the matrix
            words_by_article.append(stemmed_words)

    # making sparse matrix
    matrix = scipy.sparse.lil_matrix((len(dictionary), articles_num), dtype=float)

    # filling matrix
    for i in range(articles_num):
        for word in words_by_article[i]:
            matrix[dictionary[word], i] += 1

    # converting to csr matrix for faster operations
    matrix = scipy.sparse.csr_matrix(matrix)

    return matrix, dictionary


def multiply_by_inverse_document_frequency(matrix):
    """ Multiplying every row of matrix by its inverse document frequency. """

    matrix = scipy.sparse.csr_matrix.toarray(matrix)

    for word_inx in range(matrix.shape[0]):
        articles_with_word = len(matrix[word_inx].nonzero()[0])
        idf = math.log(articles_num / articles_with_word)
        matrix[word_inx] *= idf

    return scipy.sparse.csr_matrix(matrix)


def remove_noise(matrix, k):
    """ Removing noise by singular value decomposition and low rank approximation. """
    matrix = scipy.sparse.csc_matrix(matrix)
    u, s, vt = svds(matrix, k)

    # it will be numpy array
    new_matrix = u @ np.diag(s) @ vt

    return new_matrix


def get_matrix_and_dictionary(use_idf=False, reduce_noise=False, k=1):
    """ Getting term-by-document matrix for searches and dictionary with all words.

    Parameters:
    use_idf - whether multiply the matrix by inverse document frequency
    use_approximation - if reduce noise by singular value decomposition
                        and low rank approximation
    k - number of singular values for low rank approximation
    """

    matrix, dictionary = make_dictionary_and_matrix()

    if use_idf:
        matrix = multiply_by_inverse_document_frequency(matrix)
    else:
        matrix = scipy.sparse.csr_matrix.toarray(matrix)

    if reduce_noise:
        matrix = remove_noise(matrix, k)
    elif type(matrix) is not np.ndarray:
        matrix = scipy.sparse.csr_matrix.toarray(matrix)

    # normalization using numpy array
    for column in range(matrix.shape[1]):
        norm = np.linalg.norm(matrix[:, column])
        if norm != 0:
            matrix[:, column] /= norm

    # converting to sparse matrix for future operations
    matrix = scipy.sparse.csc_matrix(matrix)

    return matrix, dictionary
