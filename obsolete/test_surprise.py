import unittest
import numpy as np
import scipy
from sentence_transformers import SentenceTransformer


class MyTestCase(unittest.TestCase):

    def test_sematic_surprise(self):
        # 1. Load a pre-trained sentence transformer (Hugging Face)
        model = SentenceTransformer("all-MiniLM-L6-v2")  #

        line1 = "the quick brown fox jumps over the lazy dog"
        line2 = "a fast brown fox jumps over the sleepy hound"

        # 2. Convert text to high-dimensional semantic embeddings
        emb1 = model.encode(line1)
        emb2 = model.encode(line2)

        # 3. Approximate as probability distributions (e.g., using Softmax)
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        p = softmax(emb1)
        q = softmax(emb2)

        # 4. Calculate KL Divergence
        kl_divergence = scipy.stats.entropy(pk=p, qk=q)
        print(f"KL Divergence (Semantic): {kl_divergence:.4f}")

    def test_surprise(self):

        # 1. Define your two lines of text
        testing_line = "the quick brown fox jumps over the lazy dog"
        training_line = "a fast brown fox jumps over the sleepy hound"
        future_line = "I need more cookies I need more cookies"

        # 2. Tokenize and create word frequency dictionaries
        kl_divergence=calcualate_surprize_between_two_lines(training_line,training_line)
        self.assertEqual(kl_divergence, 0)

        kl_divergence = calcualate_surprize_between_two_lines(training_line, testing_line)
        self.assertEqual(round(kl_divergence,2), 7.83)

        kl_divergence = calcualate_surprize_between_two_lines(training_line, future_line )
        self.assertEqual(round(kl_divergence,2), 23.83)


# add assertion here

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


#this is equal to  scipy.stats.entropy
# surprise = calculate_surprise(actual_distribution, incoming_data)
# kl_divergence = scipy.stats.entropy(pk=actual_distribution, qk= incoming_data)
def calculate_surprise(p, q, epsilon=1e-10):
    # Convert to arrays and normalize to sum to 1
    p = np.asarray(p, dtype=float)
    p = p / np.sum(p)

    q = np.asarray(q, dtype=float)
    q = q / np.sum(q)

    # Add epsilon to prevent division by zero or log(0)
    p = np.where(p == 0, epsilon, p)
    q = np.where(q == 0, epsilon, q)

    # Re-normalize after applying epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Calculate KL Divergence: P * log(P / Q)
    divergence = np.sum(p * np.log(p / q))
    return divergence


if __name__ == '__main__':
    unittest.main()
