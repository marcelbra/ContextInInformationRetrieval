import torch
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Dict, List
from torch import Tensor
from mpl_toolkits.mplot3d import Axes3D

urls = ["https://www.nytimes.com/2020/08/06/us/crayola-masks-kids.html",
		"https://www.nytimes.com/aponline/2020/08/25/business/ap-eu-virus-outbreak-britain.html",
		"https://www.nytimes.com/article/face-shield-mask-california-coronavirus.html",
		"https://www.nytimes.com/2020/07/27/health/coronavirus-mask-protection.html",
		"https://www.nytimes.com/interactive/2020/health/coronavirus-best-face-masks.html",
		"https://www.nytimes.com/2020/05/24/health/coronavirus-face-shields.html",
		"https://www.nytimes.com/2020/07/27/business/fashion-masks-coronavirus.html",
		"https://www.nytimes.com/2020/08/14/parenting/schools-reopening-south-korea.html",
		"https://www.nytimes.com/2020/06/23/well/family/children-masks-coronavirus.html",
		"https://www.nytimes.com/2020/04/09/parenting/coronavirus-kids-masks.html"]

model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# create and embed corpus
corpus = []
for url in urls:
    article = Article(url)
    try:
        article.download()
        article.parse()
    except:
        continue
    corpus.append(article.text)
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
query_embedding = model.encode("should my child wear a face mask", convert_to_tensor=True)


def search(corpus_embeddings: Tensor,
           query_embedding:Tensor,
           top_k: int=5
           ) -> List[int]:
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

search(corpus_embeddings, query_embedding, 5)

# query corpus with the topic
school_text = "A primary school, junior school, elementary school or grade school is a school for children from about four to eleven years old, in which they receive primary or elementary education. It can refer to both the physical structure (buildings) and the organisation. Typically it comes after preschool, and before secondary school."
school_embedding = model.encode(school_text, convert_to_tensor=True)
new_corpus = []
for item_index in top_results:
    new_corpus.append(corpus[int(item_index)])
new_corpus_embeddings = model.encode(new_corpus, convert_to_tensor=True)
new_cos_scores = util.pytorch_cos_sim(school_embedding, new_corpus_embeddings)[0]


# visualize
pca = PCA(n_components=3)
embeddings = corpus_embeddings + query_embedding + school_embedding
emb = torch.cat((corpus_embeddings, query_embedding.reshape(1, 768), school_embedding.reshape(1, 768)), dim=0)


pc = pca.fit_transform(emb)
x, y, z = pc[:,0], pc[:,1], pc[:,2]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(12):
    if i == 10: # this is the query
        ax.scatter(x[i], y[i], z[i], c='g', marker='o')
    elif i == 11: # this is the school embedding
        ax.scatter(x[i], y[i], z[i], c='b', marker='o')
    else:
        ax.scatter(x[i], y[i], z[i], c='r', marker='o')

plt.show()

