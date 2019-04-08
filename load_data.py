# By Youssef
import os
import random
import numpy as np


# Load Dataset :
def load_imb_sentiment_analysis_dataset(data_path, seed=123):
    """Loads thee IMDb movie reviews sentiment analysis dataset

    # Arguments
        data_path: string, path to the data directoryself.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 25000
        Number of test samples : 25000
        Number of categories: 2 (0 - negative, 1 - positive)

    # References
        Mass et al., https://www.aclweb.org/anthology/P11-1015

        Download and uncompress archive from:
        http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    """

    imdb_data_path = os.path.join(data_path, 'aclImdb')

    # Load the training data :
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)

    # Load the testing data :
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith(".txt"):
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)

    # Shuffle the training data and labels :
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return (    (train_texts, np.array(train_labels)),
                (test_texts, np.array(test_labels)))

# My testing :
# (train_texts, train_labels), (test_texts, test_labels) = load_imb_sentiment_analysis_dataset("data/")
#
# print ("Text : " + test_texts[2])
# print ("Label : " + str(test_labels[2]))

# explore_data.plot_sample_length_distribution(test_texts)
# explore_data.plot_class_distribution(test_labels)
# explore_data.plot_frequency_disribution_of_ngrams(train_texts)

#print(normalization.ngram_vectorize(train_texts, train_labels, test_texts)[0])
# my_model.train_ngram_model()
