'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
Ghailan Fadah
CS 251 Data Analysis Visualization, Spring 2020
'''
import operator
from pathlib import Path
from queue import Empty, PriorityQueue
import re
import os
import numpy as np


def tokenize_words(text):
    '''Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.

    This method is pre-filled for you (shouldn't require modification).
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def count_words(email_path='data/enron/'):
    '''Determine the count of each word in the entire dataset (across all emails)

    Parameters:
    -----------
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_emails: int. Total number of emails in the dataset.

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use the `tokenize_words` function above to chunk it into a list of words.
    - Update the counts of each word in the dictionary.

    Hints:
    - Check out Python functions in the os and os.path modules for walking the directory structure.
    '''

    ham_files_location = os.listdir(email_path +"ham")
    spam_files_location = os.listdir(email_path + "spam")

    dic = {}
    email_count = 0

    for file_path in ham_files_location:
        f = open(email_path + "ham/" + file_path, "r")
        email_count += 1
        text = str(f.read())
        words = tokenize_words(text)
        for word in words:
                    if dic.get(word, -1) == -1:
                        dic[word] = 1
                    else:
                        dic[word] += 1
        f.close()

    for file_path in spam_files_location:
        f = open(email_path + "spam/" + file_path, "r")
        email_count += 1
        text = str(f.read())
        words = tokenize_words(text)
        for word in words:
                    if dic.get(word, -1) == -1:
                        dic[word] = 1
                    else:
                        dic[word] += 1
        f.close()

    return dic, email_count 


def find_top_words(word_freq, num_features=200):
    '''Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    '''

    sorted_x = sorted(word_freq .items(), key=operator.itemgetter(1))

    sorted_x.reverse()
    top_words = []
    top_count = []

    for i in range(num_features):
        if len(sorted_x) != 0 :
            word, count = sorted_x.pop(0)
            top_count.append(count)
            top_words.append(word)
        else:
            break

    return top_words, top_count

    


def make_feature_vectors(top_words, num_emails, email_path='data/enron/'):
    '''Count the occurance of the top W (`num_features`) words in each individual email, turn into
    a feature vector of counts.

    Parameters:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    num_emails: int. Total number of emails in the dataset.
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    feats. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Count the occurance of each word, ONLY THOSE THAT APPEAR IN `top_words`.

    HINTS:
    - Start with your code in `count_words` and modify as needed.
    '''

    '''
    num_emails = num_emails
    ham_files_location = os.listdir(email_path +"ham")
    spam_files_location = os.listdir(email_path + "spam")

    ham_emails = {}
    spam_emails = {}

    kind =[]

    ham_list = []
    spam_list = []

    data = np.empty((0, len(top_words)), np.float16)

    for file_path in ham_files_location:
        f = open(email_path + "ham/" + file_path, "r")
        text = str(f.read())
        words = tokenize_words(text)

        for word in words:
                  
                    if ham_emails.get(word, -1) == -1:
                        ham_emails[word] = 1
                    else:
                        ham_emails[word] += 1

        for x in top_words:
            if x in ham_emails:
                ham_list.append(ham_emails.get(x))
            else:
                ham_list.append(0)

        ham_arr =np.array(ham_list, np.float16)
                
        
        data = np.vstack((data, ham_arr))
        kind.append(0)
        ham_emails.clear()
        ham_list.clear()


    for file_path in spam_files_location:
        f = open(email_path + "spam/" + file_path, "r")
        text = str(f.read())
        words = tokenize_words(text)
        for word in words:

                    if spam_emails.get(word, -1) == -1:
                        spam_emails[word] = 1
                    else:
                        spam_emails[word] += 1

        for x in top_words:
            if x in spam_emails:
                spam_list.append(spam_emails.get(x))
            else:
                spam_list.append(0)

        spam_arr =np.array(spam_list, np.float16)
                
        
        data = np.vstack((data, spam_arr))
        kind.append(1)
        spam_emails.clear()
        spam_list.clear()

    return data, np.array(kind, np.float16)
    '''

    num_features = len(top_words)
    top_words_idx = {}

    for i, word in enumerate(top_words):
        top_words_idx[word] = i
    feats = np.zeros((num_emails,num_features))
    y = np.zeros((num_emails,))


    p = Path(email_path)
    sample_idx = 0

    for d in p.iterdir():
        if d.is_dir():
            if len(d.name) >= 4 and d.name[-4:] == "spam":
                spam_class = 0
            else:
                spam_class = 1

            for f in d.glob('*.txt'):
                fobj = open(f)
                txt = fobj.read()
                fobj.close()
                words = tokenize_words(txt)
                for word in words:
                    if word in top_words_idx:
                        feats[sample_idx,top_words_idx[word]]+=1

                y[sample_idx]=spam_class
                sample_idx+=1
                
    return feats, y
    pass


def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    '''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1. What proportion of the dataset samples should we use
        for the test set? e.g. 0.2 means 20% of samples are used for the test set, the remaining
        80% are used in training.
    shuffle: boolean. Should we shuffle the data before splitting it into train/test sets?

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    '''
    inds = np.arange(y.size)
    if shuffle:
        features = features.copy()
        y = y.copy()

        inds = np.arange(y.size)
        np.random.shuffle(inds)
        features = features[inds]
        y = y[inds]


    length = features.shape[0] * (1 - test_prop)
    length = int(length)
    x_train = features[0 : length, :]
    inds_train = inds[0 : length,]
    y_train = y[0 :length]
    x_test = features[length:, :]
    inds_test = inds[length:,]
    y_test = y[length:]

        
    return x_train, y_train, inds_train, x_test, y_test,inds_test


    ## ASK ABOUT INDCIES

    # Your code here:


def retrieve_emails(inds, email_path='data/enron'):
    '''Obtain the text of emails at the indices `inds` in the dataset.

    Parameters:
    -----------
    inds: ndarray of nonnegative ints. shape=(num_inds,).
        The number of ints is user-selected and indices are counted from 0 to num_emails-1
        (counting does NOT reset when switching to emails of another class).
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    Python list of str. len = num_inds = len(inds).
        Strings of entire raw emails at the indices in `inds`
    '''

    ##### NOT Required
    
    pass
