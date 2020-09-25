"""
The first experiment described in 5.1 of my bachelor's thesis.

Run on command line:
pip install newspaper3k
pip install torch torchvision
pip install sentence-transformers
pip install -U scikit-learn
pip install numpy
pip install matplotlib
"""

import torch
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List, Tuple
from torch import Tensor
import numpy as np

def get_corpus_from_links(urls: List[str]) -> Tuple[List[str], List[str], Tensor]:
    """Retrieves the corpus from the web and returns it
    together with its titles and embeddings."""
    corpus, titles = [], []
    for url in urls:
        article = Article(url)
        try:
            article.download()
            article.parse()
        except:
            continue
        corpus.append(article.text)
        titles.append(article.title)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    return titles, corpus, corpus_embeddings

def retrieve(corpus_embeddings: Tensor,
           query_embedding: Tensor,
           top_k: int=5
           ) -> List[int]:
    """Searches the corpus given the query and returns
    the top k results as indices."""
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
    return top_results.tolist()

def rerank(corpus_embeddings: Tensor,
                topic_embedding: Tensor,
                curr_top_indices: List[int],
                top_k: int = 5
                ) -> List[int]:
    new_embeddings = None
    for item_index in top_indices:
        if new_embeddings == None:
            new_embeddings = corpus_embeddings[int(item_index)].reshape(1, 768)
        else:
            new_embeddings = torch.cat((new_embeddings, corpus_embeddings[int(item_index)].reshape(1, 768)), dim=0)

    new_scores = util.pytorch_cos_sim(topic_embedding, new_embeddings)[0]
    new_scores = new_scores.cpu()
    top_results = np.argpartition(-new_scores, range(top_k))[0:top_k]
    order = [curr_top_indices[x] for x in top_results]
    return order

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
urls = ["https://www.nytimes.com/2020/08/06/us/crayola-masks-kids.html",
		"https://www.nytimes.com/article/face-shield-mask-california-coronavirus.html",
		"https://www.nytimes.com/2020/07/27/health/coronavirus-mask-protection.html",
		"https://www.nytimes.com/interactive/2020/health/coronavirus-best-face-masks.html",
		"https://www.nytimes.com/2020/05/24/health/coronavirus-face-shields.html",
		"https://www.nytimes.com/2020/07/27/business/fashion-masks-coronavirus.html",
		"https://www.nytimes.com/2020/08/14/parenting/schools-reopening-south-korea.html",
		"https://www.nytimes.com/2020/06/23/well/family/children-masks-coronavirus.html",
		"https://www.nytimes.com/2020/04/09/parenting/coronavirus-kids-masks.html"]

# Search the corpus with the querys and retrieve top k results
query = "should my child wear a face mask at school"
query_embedding = model.encode(query, convert_to_tensor=True)
titles, corpus, corpus_embeddings = get_corpus_from_links(urls)
top_indices = retrieve(corpus_embeddings, query_embedding, top_k=5)

# Re-rank the results with the context
school_text = "primary school"
school_embedding = model.encode(school_text, convert_to_tensor=True)
top_indices = rerank(corpus_embeddings, school_embedding, top_indices)

# Visualisation
embeddings = torch.cat((corpus_embeddings, query_embedding.reshape(1, 768), school_embedding.reshape(1, 768)), dim=0).cpu()
pc = PCA(n_components=2).fit_transform(embeddings)
x, y= pc[:,0], pc[:,1]
fig, ax = plt.subplots()
corpus_amount = corpus_embeddings.size()[0]
embeds_amount = embeddings.size()[0]
for i in range(embeds_amount):
    # The corpus
    if i < corpus_amount:
        ax.scatter(x[i], y[i], c='g', marker='o')
        ax.annotate(str(i + 1), (x[i] + 0.5, y[i] - 0.25 ))
    # The query
    elif i == corpus_amount:
        ax.scatter(x[i], y[i], c='b', marker='o')
        ax.annotate("General Query", (x[i] + 0.5, y[i] - 0.25 ))
    elif i == corpus_amount + 1:
        ax.scatter(x[i], y[i], c='r', marker='o')
        ax.annotate("School", (x[i] + 0.5, y[i] - 0.25 ))
plt.axis('off')
plt.show()