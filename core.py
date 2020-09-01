from corpus import Corpus
from document import Document
from typing import List, Dict, Union, Tuple
class IRCore:

    def __init__(self,
                 corpus: Corpus=None):
        self.corpus = corpus

    def search(self, query: str) -> List[Document]:
        pass