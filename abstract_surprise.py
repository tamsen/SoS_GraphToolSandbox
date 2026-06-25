import os
import unittest
from pathlib import Path

import numpy as np
import scipy
#from sentence_transformers import SentenceTransformer


def calculate_surprise():

    #data_dir="/Users/tdunn/Data/SoS"
    data_dir = Path(__file__).parent
    abstract_t0=os.path.join(data_dir,"data","Abstract_t0.txt")
    abstract_t1=os.path.join(data_dir,"data","Abstract_t1.txt")

    testing_line = "the quick brown fox jumps over the lazy dog"
    training_line = "a fast brown fox jumps over the sleepy hound"
    future_line = "I need more cookies I need more cookies"

    with open(abstract_t1, 'r') as file:
        content_1 = file.read()

    with open(abstract_t0, 'r') as file:
        content_0 = file.read()

    # sanity check, this should be zero
    kl_divergence=calcualate_surprize_between_two_lines(content_1,content_1)

    #how surprising is content_1 given content_0?
    kl_divergence = calcualate_surprize_between_two_lines(content_0,content_1)

    kl_divergence = calcualate_surprize_between_two_lines(training_line, future_line )


def calcualate_surprize_between_two_lines(training_line2,testing_line1,):
    words1 = testing_line1.split()
    words2 = training_line2.split()
    all_words = list(set(words1 + words2))

    epsilon = 1e-10
    p = np.array([words1.count(w) for w in all_words], dtype=float)
    p += epsilon
    p /= np.sum(p)  # Normalize to sum to 1

    q = np.array([words2.count(w) for w in all_words], dtype=float)
    q += epsilon
    q /= np.sum(q)  # Normalize to sum to 1

    # 4. Calculate KL divergence using scipy
    # KL(p || q): Surprise of expecting line2 but seeing line1
    kl_divergence = scipy.stats.entropy(pk=p, qk=q)
    print(f"KL Divergence (surprise): {kl_divergence:.4f}")
    return kl_divergence



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    calculate_surprise()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
