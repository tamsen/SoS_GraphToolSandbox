import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import json
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from sentence_transformers import SentenceTransformer


def calculate_surprise():

    data_dir = Path(__file__).parent
    abstract_t0=os.path.join(data_dir,"data","Abstract_t0.txt") #past abstract
    abstract_t1=os.path.join(data_dir,"data","Abstract_t1.txt") #present abstract
    abstract_t2 = os.path.join(data_dir, "data", "Abstract_t2.txt")  #future abstract

    ThreeBreakthroughPaper_References_Citations_Path = (
        os.path.join(data_dir, "data", "ThreeBreakthroughPaper_References_Citations.txt"))  # future abstract
    RandomPaper_References_Citations_Path = (
        os.path.join(data_dir, "data", "YEARRandomPaper_References_Citations.txt"))  # future abstract

    Paperid_Title_Abstract_Path = (
        os.path.join(data_dir, "data", "Paperid_Title_Abstract.txt"))  # future abstract

    InformationTheoryCases = {
        'W2141394518': 'Lorenz 1963',
        'W2126466006': 'DNA 1953',
        'W2126160338': 'Turing 1936'
    }

    ThreeBreakthroughs_R = {}
    ThreeBreakthroughs_C = {}
    with open(ThreeBreakthroughPaper_References_Citations_Path, 'r') as f:
        for line in f:
            line = json.loads(line.strip('\n'))
            ThreeBreakthroughs_R[line['id']] = line['references']
            ThreeBreakthroughs_C[line['id']] = line['citations']

    RandomPaper_R = {}
    RandomPaper_C = {}
    RandomPaper_Y = {}
    for y in [1936, 1953, 1963]:
        with open(RandomPaper_References_Citations_Path.replace("YEAR",str(y)), 'r') as f:
            for line in f:
                line = json.loads(line.strip('\n'))
                RandomPaper_R[line['id']] = line['references']
                RandomPaper_C[line['id']] = line['citations']
                RandomPaper_Y[line['id']] = y

    PaperTitle = {}
    PaperAbstract = {}
    with open(Paperid_Title_Abstract_Path, 'r') as f:
        for line in f:
            line = json.loads(line.strip('\n'))
            PaperTitle[line['id']] = line['title']
            PaperAbstract[line['id']] = line['abstract_inverted_index']

    #start with 'Lorenz 1963'
    Lorenz_cits = ThreeBreakthroughs_C['W2141394518']
    Lorenz_refs = ThreeBreakthroughs_R['W2141394518']

    # make a list of all the words in the abstracts of lorenz citations
    content_0_words=[]
    for citation in Lorenz_refs:
        print(citation)
        content_0_words = content_0_words+list(PaperAbstract["W105176998"].keys())

    # make a list of all the words in the lorenz abstract
    content_1_words = []
    for citation in Lorenz_cits:
        content_1_words = content_1_words + list(PaperAbstract[citation].keys())

    # make a list of all the words in the abstracts of lorenz references
    content_2_words=[]
    for citation in Lorenz_cits:
        content_2_words = content_2_words + list(PaperAbstract[citation].keys())


    # sanity check, this should be zero
    kl_divergence_should_be_zero= calculate_surprise_between_two_word_lists(
        content_0_words, content_0_words)

    #how surprising is content_1 given content_0?
    Novelty_of_abs_1_given_abs_0 = calculate_surprise_between_two_word_lists(content_0_words,
                                                                             content_1_words)

    #how surprising is content_2 given content_1?
    Novelty_of_abs_2_given_abs_1 = calculate_surprise_between_two_word_lists(content_1_words,
                                                                             content_2_words)
    Novelty_of_abs_2_given_abs_0 = calculate_surprise_between_two_word_lists(content_0_words,
                                                                             content_2_words)

    #Concepts of Transience & Resonance from
    # "Individuals, institutions, and innovation in the debates
    #        of the French Revolution" DeDeo paper
    #Note: KL divergence can be calculated severale ways.
    # "Calculate_sematic_surprise_between_two_word_lists" is better but more expensive
    # "calculate_sematic_surprise_between_two_lines" is cheap and goes by pure probability

    # Novelty: surprise of now, given the past  (how much now is not like the past)
    # Transience: surprise of now, given the future (how much now is not like the future)
    Transience_of_abs_1 = calculate_surprise_between_two_word_lists(content_2_words, content_1_words)

    # Resonance is novelty minus transience
    Resonance_of_abs_1 = Novelty_of_abs_1_given_abs_0-Transience_of_abs_1
    print("Resonance_of_abs_1:",Resonance_of_abs_1)

    #Impact. Is Novelty(2|1) < Novelty(2|0) ?
    #If  abs 0 predicts the future just as well as abs 1, then abs 1 didnt have much impact!
    Impact_of_abs_1 = Novelty_of_abs_2_given_abs_0 - Novelty_of_abs_2_given_abs_1
    print("Impact_of_abs_1:",Impact_of_abs_1)


def calculate_surprise_between_two_word_lists(training_words, testing_words, ):

    all_words = list(set(training_words + testing_words))

    epsilon = 1e-10
    p = np.array([testing_words.count(w) for w in all_words], dtype=float)
    p += epsilon
    p /= np.sum(p)  # Normalize to sum to 1

    q = np.array([training_words.count(w) for w in all_words], dtype=float)
    q += epsilon
    q /= np.sum(q)  # Normalize to sum to 1

    # 4. Calculate Kullback–Leibler divergence using scipy
    # KL(p || q): Surprise of expecting line2 but seeing line1
    kl_divergence = scipy.stats.entropy(pk=p, qk=q)
    print(f"KL Divergence (surprise): {kl_divergence:.4f}")
    return kl_divergence


def calculate_sematic_surprise_between_two_lines(testing_line, training_line):

    # 1. Load a pre-trained sentence transformer (Hugging Face)
    model = SentenceTransformer("all-MiniLM-L6-v2")  #

    # 2. Convert text to high-dimensional semantic embeddings
    emb1 = model.encode(testing_line)
    emb2 = model.encode(training_line)

    # 3. Approximate as probability distributions (e.g., using Softmax)
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    p = softmax(emb1)
    q = softmax(emb2)

    # 4. Calculate KL Divergence
    kl_divergence = scipy.stats.entropy(pk=p, qk=q)
    print(f"KL Divergence (Semantic): {kl_divergence:.4f}")


if __name__ == '__main__':
    calculate_surprise()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
