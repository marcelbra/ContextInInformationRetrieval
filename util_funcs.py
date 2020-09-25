import torch
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple, Optional, Union
from torch import Tensor
import numpy as np

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')#distilbert-base-nli-mean-tokens')

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
        corpus.append(filter_corona_words(article.text))
        titles.append(filter_corona_words(article.title))
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    return titles, corpus, corpus_embeddings

def get_embeddings_from_corpus(corpus: Union[List[str], str]) -> Tensor:
    if isinstance(corpus, str):
        return model.encode(corpus, convert_to_tensor=True).reshape(1, 768)
    else:
        return model.encode(corpus, convert_to_tensor=True)

def filter_corona_words(doc: str) -> str:
    # doc = doc.replace("", "")
    return doc

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
    for item_index in curr_top_indices:
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
    return model.encode(x, convert_to_tensor=True)
    # en = lambda x: model.encode(x)