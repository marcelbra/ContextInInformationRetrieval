"""


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

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

urls = [
    "https://www.nytimes.com/2020/03/13/world/how-to-wash-your-hands-coronavirus.html",
    "https://www.nytimes.com/2020/06/11/well/live/the-hand-washing-wars.html",
    "https://www.nytimes.com/interactive/2020/04/15/burst/how-to-wash-your-hands.html",
    "https://www.nytimes.com/2016/04/21/health/washing-hands.html"
    ]

wiki = [
    "Hand washing (or handwashing), also known as hand hygiene, is the act of cleaning one's hands with soap (or equivalent materials) and water to remove viruses/bacteria/microorganisms, dirt, grease, or other harmful and unwanted substances stuck to the hands. Drying of the washed hands is part of the process as wet and moist hands are more easily recontaminated.", # hand washing -> first paragraph
    "Hand washing has many significant health benefits, including minimizing the spread of influenza, coronavirus, and other infectious diseases; preventing infectious causes of diarrhea; decreasing respiratory infections; and reducing infant mortality rate at home birth deliveries. A 2013 study showed that improved hand washing practices may lead to small improvements in the length growth in children under five years of age.", # hand washing -> public health -> healt benefits -> first paragraph
    "When not wearing a mask, the CDC, WHO, and NHS recommends covering the mouth and nose with a tissue when coughing or sneezing and recommends using the inside of the elbow if no tissue is available.[112][121][132] Proper hand hygiene after any cough or sneeze is encouraged.[112][121] The WHO also recommends that individuals wash hands often with soap and water for at least 20 seconds, especially after going to the toilet or when hands are visibly dirty, before eating and after blowing one's nose." # covid-19 pandemic -> hand washing -> first paragraph
    ]

paragraphs = [
    "Hand washing cleans the hand with soap to remove harmful viruses, bacteria and microorganisms." # compressed first wiki paragraph
    "Hand washing removes viruses from the hand." # more concise paragraph with emphasis on virus on hands
    "Hand washing is an effective hygiene measure in a pandemic" # more conside paragraph with emphasis is on measure in pandemic
    ]

topics = [
    "hand washing",
    "hand cleaning"
    "washing hands"
    "wash hands"
    "disinfect hands"
    ]

emb_corpus = get_corpus_from_links(urls)[2]
emb_wiki = get_embeddings_from_corpus(wiki)
emb_para = get_embeddings_from_corpus(paragraphs)
emb_topics = get_embeddings_from_corpus(topics)
embeddings = torch.cat((emb_corpus, emb_wiki, emb_para, emb_topics), dim=0).cpu()
