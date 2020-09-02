import torch
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from torch import Tensor
from context_concepts import contexts
from mpl_toolkits.mplot3d import Axes3D

def get_corpus_from_links(urls: List[str]) -> Tuple[List[str], Tensor]:
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
    return corpus, corpus_embeddings

def filter(corpus_embeddings: Tensor,
           query_embedding: Tensor,
           top_k: int=5
           ) -> List[int]:
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
    return top_results.tolist()

def new_order(l: List[int], mirror: List[int]) -> List[int]:
    return [l[x] for x in mirror]

def post_filter(corpus_embeddings: Tensor,
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
    top_results = np.argpartition(-new_scores, range(top_k))[0:top_k]
    order = new_order(curr_top_indices, top_results)
    return order

# link collection
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

# filter corpus with query and retrieve top k indices
query_embedding = model.encode("should my child wear a face mask", convert_to_tensor=True)
corpus, corpus_embeddings = get_corpus_from_links(urls)
top_indices = filter(corpus_embeddings, query_embedding, 5)

# query corpus with the topic
school_text = "A primary school, junior school, elementary school or grade school is a school for children from about four to eleven years old, in which they receive primary or elementary education. It can refer to both the physical structure (buildings) and the organisation. Typically it comes after preschool, and before secondary school."
school_embedding = model.encode(school_text, convert_to_tensor=True)
top_indices = post_filter(corpus_embeddings, school_embedding, top_indices)


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

title = "Social distancing"
text = "Social distancing (also known as physical distancing) includes infection control actions intended to slow the spread of disease by minimising close contact between individuals. Methods include quarantines; travel restrictions; and the closing of schools, workplaces, stadiums, theatres, or shopping centres. Individuals may apply social distancing methods by staying at home, limiting travel, avoiding crowded areas, using no-contact greetings, and physically distancing themselves from others. Many governments are now mandating or recommending social distancing in regions affected by the outbreak. Non-cooperation with distancing measures in some areas has contributed to the further spread of the pandemic"
emb1 = model.encode(title)
emb2 = model.encode(text)
cos_scores = util.pytorch_cos_sim(emb1, emb2)[0]
s = 0