from torch import Tensor
from typing import List, Dict

class Topic:

    def __init__(self,
                 text: str=None,
                 embedding: Tensor=None):
        self.text = text
        self.embedding = embedding