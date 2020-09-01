from torch import Tensor

class Document(object):

    def __init__(self,
                 name: str,
                 text:str,
                 embedding: Tensor):
        self.name = name
        self.text = text
        self.embedding = embedding
