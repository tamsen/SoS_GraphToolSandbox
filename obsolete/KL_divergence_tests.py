import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy

# 1. Define two text documents
text1 = "the quick brown fox jumps over the lazy dog"
text2 = "a fast brown fox jumps over the lazy dog"

# 2. Vectorize the text to get word counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([text1, text2]).toarray()

# 3. Convert counts to probability distributions
p = X[0].astype(float)
q = X[1].astype(float)

p = p / np.sum(p)
q = q / np.sum(q)

# 4. Smooth to handle zero probabilities
epsilon = 1e-10
p = p + epsilon
q = q + epsilon

# Re-normalize to ensure they sum to 1
p = p / np.sum(p)
q = q / np.sum(q)

# 5. Compute KL Divergence (P || Q)
# Note: In SciPy, passing qk computes the KL divergence of pk relative to qk
kl_div = entropy(pk=p, qk=q)
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html
print(f"KL Divergence: {kl_div:.4f}")


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


# True distribution vs. New/Surprising data
actual_distribution = [0.05, 0.80, 0.15]
incoming_data = [0.50, 0.20, 0.30]

surprise = calculate_surprise(actual_distribution, incoming_data)
print(f"Smoothed Surprise Rate: {surprise:.4f}")