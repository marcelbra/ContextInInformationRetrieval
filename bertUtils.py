import torch
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from torch import Tensor
from context_concepts import contexts
import numpy as np
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
    cos_scores = cos_scores.cpu()
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
    new_scores = new_scores.cpu()
    top_results = np.argpartition(-new_scores, range(top_k))[0:top_k]
    order = new_order(curr_top_indices, top_results)
    return order

def similarity_matrix(encoded_topics: List[Tensor]
                      ) -> List[List[float]]:
    topic_length = len(encoded_topics)
    m = [[0] * topic_length for x in range(topic_length)]
    for i, row in enumerate(encoded_topics):
        for j, col in enumerate(encoded_topics):
            m[i][j] = round(float(dist(row, col)), 4)
    return m

def dist(x, y):
    return util.pytorch_cos_sim(x,y)[0]
    # dist = lambda x, y: util.pytorch_cos_sim(x,y)[0]

def dist_w(x, y):
    return util.pytorch_cos_sim(en(x),en(y))[0]
    # dist_w = lambda x, y: util.pytorch_cos_sim(en(x),en(y))[0]
def en(x):
    return model.encode(x)
    # en = lambda x: model.encode(x)

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')#distilbert-base-nli-mean-tokens')
"""
Experiment 001

Setting:
Taking a toy corpus of 10 corona articles. All articles talk about
masks and some talk about masks w.r.t. kids and their environment (school).
We assume a mother (role) that wants to know if her child should wear a mask (e.g. in
school (location)). The location will be context modelled here (i.e. encoded through a wiki article)
and and used in a post-processing manner.

Hypothesis:
The hypothesis is that having this knowledge about the mother and the setting
she is interested in results in better information.

Result:
We can see that when we query we retrieve as results the ranking [5,4,0,9,8]. This shows that
the general face mask articles are ranked 1 and 2. The third article is a kids article, however
the two best articles which explicitly talk about what the mother wants to know are ranked low.
After querying against that ranking with the school we retrieve a new ranking [0, 8, 9, 5, 4]
which, without a doubt, would perfectly meet the mother's concern.

The PCA visualizes that the documents indeed are very close to each other. It shows that in this
semantic space only minor differences are responsible for worse or better results.
"""
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
# filter corpus with query and retrieve top k indices
query_embedding = model.encode("should my child wear a face mask", convert_to_tensor=True)
corpus, corpus_embeddings = get_corpus_from_links(urls)
top_indices = filter(corpus_embeddings, query_embedding, 5)
# query corpus with the topic
school_text = "child"#A primary school, junior school, elementary school or grade school is a school for children from about four to eleven years old, in which they receive primary or elementary education. It can refer to both the physical structure (buildings) and the organisation. Typically it comes after preschool, and before secondary school."
school_embedding = model.encode(school_text, convert_to_tensor=True)
top_indices = post_filter(corpus_embeddings, school_embedding, top_indices)
# visualisation
pca = PCA(n_components=3)
embeddings = corpus_embeddings + query_embedding + school_embedding
emb = torch.cat((corpus_embeddings, query_embedding.reshape(1, 768), school_embedding.reshape(1, 768)), dim=0)
pc = pca.fit_transform(emb.cpu())
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
# plt.show()

"""
Experiment 002

Setting:
We now look at the context "theme" and one of its topics "mask".
As we are in latent high dimensional space we cannot make sense of representation of encoded words.
Moreover, making sense of encoded sentences or documents cannot be made sense of even more.
It is crucial to verify if encoded information are precise and if words, descriptions and texts that
are supposedly close to each are so indeed.

Hypothesis:
The hypothesis is that encoded wikipedia articles comprise important knowledge and are a better representation
of a certain concept than just a word alone.

Results:
- The two texts describing the topic in the wiki article are VERY close to each other (0.8398)
  - This implies that the underlying topic, despite talking about different parts of it, is encoded in the vectors
- Two first section seems to be closer to the other words
  - This implies that choosing the right context to embed is of importance w.r.t. accuracy
- The custom, supposedly very precise, description of a the corona face mask, is very close to all concepts
  - This implies that embedding the text might be reasonable
- The custom, precise description is very close to the wikipedia articles, not as close to general masks
  - This implies that these sections to cover semantic knowledge about the circumstances without being trained on COVID
- Facemask and Face mask alone have almost no similary to the wiki articles
  - This implies that embedding the word alone may not be precise enough
  
Matrix:
[1.0, 0.8398, 0.1179, 0.1427, 0.3443, 0.1397, 0.6104]
[0.8398, 1.0, 0.0, 0.0514, 0.3142, 0.0476, 0.5565]
[0.1179, 0.0, 1.0, 0.9056, 0.781, 0.8695, 0.4708]
[0.1427, 0.0514, 0.9056, 1.0, 0.8334, 0.9466, 0.4629]
[0.3443, 0.3142, 0.781, 0.8334, 1.0, 0.8444, 0.5279]
[0.1397, 0.0476, 0.8695, 0.9466, 0.8444, 1.0, 0.4656]
[0.6104, 0.5565, 0.4708, 0.4629, 0.5279, 0.4656, 1.0]
"""
fm1 = en(contexts["theme"]["face mask"]["text"][0])
fm2 = en(contexts["theme"]["face mask"]["text"][1])
fm3 = en("facemask")
fm4 = en("face mask")
fm5 = en("surgical mask")
fm6 = en("mask")
fm7 = en("mouth-nose-covering used to prevent viral spread")
all_words1 = [fm1, fm2, fm3, fm4, fm5, fm6, fm7]
sim_matrix1 = similarity_matrix(all_words1)
print(*sim_matrix1, sep="\n")

"""
Experiment 003

Setting:
We now look at the context "theme" and one of its topics "social distancing".

Hypothesis:
Same as before.

Results:
We can see similar results as above.
The encoded topics are all quite similar to each other.
The first encoded topic is the best performing w.r.t. the other topics.
Encoded Topic 2 and 3 have almost 0 similarity with the hardcoded encoded words.
  - This implies that there are definitely bad candidate sentences.
  - This implies that there is a need for evaluating the best performing textual representation
If we only look at the synonyms we see that all encodings are very close to each other
with a score of 0.7 - 0.9.
  - This implies that encoding a very particular word may not be that important.
Matrix:
[1.0000, 0.6898, 0.6943, 0.7919, 0.3188, 0.2143, 0.1944, 0.2310, 0.2633, 0.3341, 0.3174, 0.3163]
[0.6898, 1.0000, 0.5752, 0.7837, 0.0723, 0.0529, 0.0004, 0.0436, 0.0559, 0.0942, 0.0621, 0.0201]
[0.6943, 0.5752, 1.0000, 0.5847, 0.0905, 0.0552, 0.0821, 0.0431, 0.0429, 0.1190, 0.1124, 0.1262]
[0.7919, 0.7837, 0.5847, 1.0000, 0.3147, 0.1441, 0.2077, 0.2037, 0.2474, 0.3431, 0.3187, 0.1973]
[0.3188, 0.0723, 0.0905, 0.3147, 1.0000, 0.8677, 0.7550, 0.7949, 0.7975, 0.8075, 0.8187, 0.7211]
[0.2143, 0.0529, 0.0552, 0.1441, 0.8677, 1.0000, 0.7859, 0.8514, 0.7194, 0.8115, 0.8280, 0.7862]
[0.1944, 0.0004, 0.0821, 0.2077, 0.7550, 0.7859, 1.0000, 0.8705, 0.6632, 0.8490, 0.8751, 0.7405]
[0.2310, 0.0436, 0.0431, 0.2037, 0.7949, 0.8514, 0.8705, 1.0000, 0.7904, 0.9105, 0.9346, 0.8691]
[0.2633, 0.0559, 0.0429, 0.2474, 0.7975, 0.7194, 0.6632, 0.7904, 1.0000, 0.7319, 0.7447, 0.6668]
[0.3341, 0.0942, 0.1190, 0.3431, 0.8075, 0.8115, 0.8490, 0.9105, 0.7319, 1.0000, 0.9917, 0.8196]
[0.3174, 0.0621, 0.1124, 0.3187, 0.8187, 0.8280, 0.8751, 0.9346, 0.7447, 0.9917, 1.0000, 0.8333]
[0.3163, 0.0201, 0.1262, 0.1973, 0.7211, 0.7862, 0.7405, 0.8691, 0.6668, 0.8196, 0.8333, 1.0000]
"""
sd1 = en(contexts["theme"]["social distancing"]["text"][0])
sd2 = en(contexts["theme"]["social distancing"]["text"][1])
sd3 = en(contexts["theme"]["social distancing"]["text"][2])
sd4 = en(contexts["theme"]["social distancing"]["text"][3])
sd5 = en("social distancing")
sd6 = en("social distance")
sd7 = en("not going to public areas")
sd8 = en("avoiding physical contact")
sd9 = en("cutting oneself off")
sd10 = en("avoiding contacts with people")
sd11 = en("avoiding contacts with individuals")
sd12 = en("avoiding crowded places")
sd13 = en("A drastic measure of social distancing avoiding physical contact with other people and crowded places.")
all_words2 = [eval(f"sd{i}") for i in range(1,13)]
sim_matrix2 = similarity_matrix(all_words2)
print(*sim_matrix2, sep="\n")


"""
General Notes:
- it is of very importance to find out how to embed the topics the best way
	- embedding wikipedia articles yields different results. some are really good (w.r.t the solely embedded terms) and some are really bad. Wikipedia articles might be too general 
	for the model to capture the notion / sense behind it
	- embedding described topics by hand with precise and many synomys might be good
	- embedding the most general concept that describes the topic might be the best
- different models yield different vector spaces
- Pre-trained models only know general concepts. Corona specific words are unknown and have no sense in the vector space 
	- Possibly map specific terms to general terms
"""