"""
The second experiment described in 5.2 of my bachelor's thesis.

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
from typing import List, Tuple, Union
from torch import Tensor
import numpy as np


def retrieve(corpus_embeddings: Tensor,
             query_embedding: Tensor,
             top_k: int = 5
             ) -> List[int]:
    """Searches the corpus given the query and returns
    the top k results as indices."""
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
    return top_results.tolist()


def geometric(number: int) -> float:
    return (1/1.3)**number

def step(searcher: Tensor,
         top_indices: List[int],
         emb_measures: Tensor,
         counter: int
         ) -> Tuple[Tensor, Tensor]:
    """

    :param searcher:
    :param top_indices:
    :param emb_measures:
    :param counter:
    :return:
    """
    topics = [emb_measures[retrieve(emb_measures, emb_corpus[index], 1)[0]].reshape(1,768) for index in top_indices]
    searcher_history = searcher.clone()
    for topic in topics:
        direction = topic - searcher
        distance = geometric(counter) / len(topics)
        searcher = searcher + (direction * distance)
        searcher_history = torch.cat((searcher_history, searcher), dim=0)
    return searcher, searcher_history

def get_corpus_from_links(urls: List[str]) -> Tuple[List[str], List[str], Tensor]:
    # create and embed corpus
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

def get_embeddings_from_corpus(corpus: Union[List[str], str]) -> Tensor:
    if isinstance(corpus, str):
        return model.encode(corpus, convert_to_tensor=True).reshape(1, 768)
    else:
        return model.encode(corpus, convert_to_tensor=True)

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
urls = ["https://www.nytimes.com/2020/06/02/realestate/virus-social-distancing-etiquette-rules.html",
        "https://www.nytimes.com/2020/05/04/us/social-distancing-rules-coronavirus.html",
        "https://www.nytimes.com/2020/03/16/smarter-living/coronavirus-social-distancing.html",
        "https://www.nytimes.com/2020/06/18/nyregion/coronavirus-ny-social-distancing.html",
        "https://www.nytimes.com/2020/03/18/world/clean-home-coronavirus.html",
        "https://www.nytimes.com/2020/05/06/well/live/coronavirus-cleaning-cleaners-disinfectants-home.html",
        "https://www.nytimes.com/guides/smarterliving/how-to-clean",
        "https://www.nytimes.com/wirecutter/reviews/best-all-purpose-cleaner/",
        "https://www.nytimes.com/2020/03/13/world/how-to-wash-your-hands-coronavirus.html",
        "https://www.nytimes.com/2020/06/11/well/live/the-hand-washing-wars.html",
        "https://www.nytimes.com/interactive/2020/04/15/burst/how-to-wash-your-hands.html",
        "https://www.nytimes.com/2016/04/21/health/washing-hands.html",
        "https://www.nytimes.com/wirecutter/reviews/best-cloth-face-masks/",
        "https://www.nytimes.com/interactive/2020/07/17/upshot/coronavirus-face-mask-map.html",
        "https://www.nytimes.com/article/face-shield-mask-california-coronavirus.html",
        "https://www.nytimes.com/interactive/2020/08/10/nyregion/nyc-subway-coronavirus.html",
        "https://www.nytimes.com/2020/03/04/opinion/coronavirus-buildings.html",
        "https://www.nytimes.com/2020/07/27/health/coronavirus-mask-protection.html",
        "https://www.nytimes.com/2020/07/06/health/coronavirus-airborne-aerosols.html",
        "https://www.nytimes.com/2020/07/30/opinion/coronavirus-aerosols.html"]

colors = ["#00FFCC", "#00FFCC", "#00FFCC", "#00FFCC", # social distancing
          "#F7347A", "#F7347A", "#F7347A", "#F7347A", # surface cleaning
          "#FFFF66", "#FFFF66", "#FFFF66", "#FFFF66", # washing hands
          "#F08080", "#F08080", "#F08080", "#F08080", # face masks
          "#990000", "#990000", "#990000", "#990000"] # air circulation

# Embed context, corpus and place searcher
measures = ["social distance", "surface cleaning", "hand washing", "face masks", "air filtration"]
titles, corpus, emb_corpus = get_corpus_from_links(urls)
emb_measures = get_embeddings_from_corpus(measures)
searcher = torch.mean(emb_measures, dim=0, keepdim=True)

# We assume the user is interested in hand washing
query = "Is hand washing an effective measure against the virus?"
emb_query = get_embeddings_from_corpus(query)
top_indices = retrieve(emb_corpus, emb_query, 5)

# The top indices are [8, 11, 4, 10, 5]
# We can see that 8, 11, 10 are relevant and 4, 5 are not
# For simplicity we assume the user has chosen all 4 documents with topic handwashing

# We now perform 4 optimization steps (we pick the same topics, but it simulates picking 4 different ones)
searcher, history = step(searcher, top_indices[:2], emb_measures, 1)
searcher, history2 = step(searcher, top_indices[:2], emb_measures, 2)
history2 = history2[1:]

# Visualization of the experiment
embeddings = torch.cat((emb_measures, emb_corpus, history, history2), dim=0)
pc = PCA(n_components=2).fit_transform(embeddings.cpu())
x, y = pc[:,0], pc[:,1]
fig, ax = plt.subplots()
p = 0
corpus_size, measures_size, histo1_size, histo2_size = len(corpus), len(measures), history.size()[0], history2.size()[0]
for i in range(corpus_size + measures_size + histo1_size + histo2_size):
    # Plot measures
    if i < measures_size:
        ax.scatter(x[i], y[i], c=colors[i * (len(measures)-1)], marker='o')
        ax.annotate(measures[i], (x[i] - 3.5, y[i] + 0.5))
    # Plot corpus
    elif i < (measures_size + corpus_size): # docs
        ax.scatter(x[i], y[i], c=colors[i - len(measures)], marker='o')
        name = str(i-len(measures)+1) if i-len(measures) >= 10 else " " + str(i-len(measures)+1)
        ax.annotate(name, (x[i]-0.2, y[i]-0.2), fontsize=6)
    # Plot Searcher
    else:
        ax.scatter(x[i], y[i], c='b', marker='o')
        ax.annotate(str(p), (x[i] + 0.5, y[i] - 0.25 ))
        p += 1
plt.axis('off')
plt.show()