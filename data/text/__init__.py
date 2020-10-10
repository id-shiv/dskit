from scipy import spatial
from sent2vec.vectorizer import Vectorizer

def sen_similarity(sen1: str, sen2: str):
    """ Returns similarity between two input sentences

    :param sen1: first sentence
    :param sen2: second sentence

    :return: similarity score between 0 and 1, closer to 0 more similar.
    """

    # vectorize the sentences
    vectorizer = Vectorizer()
    vectorizer.bert([sen1, sen2])
    vectors_bert = vectorizer.vectors

    similarity = spatial.distance.cosine(vectors_bert[0], vectors_bert[1])

    return similarity
    