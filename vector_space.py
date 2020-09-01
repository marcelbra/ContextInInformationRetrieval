from typing import List, Dict
from torch import Tensor
from sentence_transformers import SentenceTransformer, util



class VectorSpace(object):

    def __init__(self):
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    def embed(self, docs: List[str]) -> Tensor:
        return self.model.encode(docs, convert_to_tensor=True)