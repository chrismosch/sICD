import numpy as np
import pandas as pd
from collections import Counter


# notation similar to S. Williamson, C. Wang, K. A. Heller and D. M. Blei. The IBP-compound Dirichlet process and its application to focused topic modeling. ICML, 2010

def IBP_stick_breaking(alpha, N, K):
    mu = np.random.beta(alpha,1,size=K)
    pi = np.cumprod(mu)
    return pi

    
def FTM_synthetic_data(M, K, Nvocab, gamma, alpha, eta, supervised=False, mu_a=0, sigma_a=1, sigma_y=1):

    phi = np.random.gamma(gamma,1, size=K)
    beta = np.random.dirichlet(eta, size=K)  # topic distribution over words

    while True:
        pi = IBP_stick_breaking(alpha,M,K)
        b = np.random.binomial(n=1, p=pi, size=(M,K))
        if np.all(b.dot(phi)>0):  # negative_binomial requires parameter n>0
            break
    while True:
        n = np.random.negative_binomial(b.dot(phi),0.5)
        if np.all(n>0):  # ignore draws where some documents consist of 0 words (alternatively: drop those cases)
            break
    theta = np.apply_along_axis(np.random.dirichlet, arr=b*phi, axis=1)

    zs,ws = [],[]
    for m in range(M):
        # draw topic for each word slot
        z = np.random.choice(range(K), p=theta[m,:], size=n[m])
        zs.append(z)
        # draw word for each word slot
        w = [np.random.choice(range(Nvocab), p=beta[z_word,:]) for z_word in z]
        ws.append(w)
    
    if supervised:
        n_per_mk = get_n_per_mk(zs,K)
        a = np.random.normal(mu_a, sigma_a, size=K)
        n_per_mk_norm = n_per_mk / n_per_mk.sum(axis=1)[:,np.newaxis]
        y = np.random.normal(n_per_mk_norm.dot(a), sigma_y)
        return phi, beta, pi, b, n, theta, zs, ws, y, a
    else:
        return phi, beta, pi, b, n, theta, zs, ws, None, None


def get_n_per_mk(z, K):
    # Nwords_per_doc_and_topic
    topic_counts_per_doc = [Counter(z_m) for z_m in z]
    n_per_mk = pd.DataFrame(topic_counts_per_doc, columns=range(K))
    n_per_mk = np.nan_to_num(n_per_mk)
    return n_per_mk


def get_n_per_ik(z, w, K, Nvocab):
    # Nvocabulary x K matrix to look up how often particular word associated with particular topic
    topic_word_pair_counts_per_doc = [Counter(zip(z_m,w_m)) for z_m,w_m in zip(z,w)]
    topic_word_pair_counts_per_doc = pd.DataFrame(topic_word_pair_counts_per_doc)
    topic_word_pair_counts_series = topic_word_pair_counts_per_doc.sum(axis=0)
    k,i = zip(*topic_word_pair_counts_series.index)
    # topic_word_pair_counts
    n_per_ik = np.zeros((Nvocab,K))
    n_per_ik[i,k] = topic_word_pair_counts_series.values
    return n_per_ik