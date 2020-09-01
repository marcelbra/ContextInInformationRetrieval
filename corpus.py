from vector_space import VectorSpace
from document import Document
from typing import Dict, List, Tuple
import torch
from torch import Tensor
from newspaper import Article

class Corpus(VectorSpace):

    def __init__(self,
                 space: Tensor=None,
                 docs: List[Document]=None):
        super(Corpus, self).__init__()
        self.space = space
        self.docs = docs

    def search(self, query: str) -> List[Document]:
        emb_query = self.embed([query])
        return [Document()]


    def load(self,
             source: str,
             mode: str,
             docs: List[str]):

        # load new docs and embeddings
        if source == "urls":
            new_embeddings, new_docs = self.load_from_urls(docs)
        elif source == "file":
            pass
        else:
            pass

        # add or override new docs and embeddings
        if not self.space or mode == "override":
            self.space = new_embeddings
            self.docs = new_docs
        elif mode == "add":
            self.space = torch.cat((self.space, new_embeddings), dim=0)
            self.docs.append(new_docs)

    def load_from_urls(self, urls: List[str]) -> Tuple[Tensor, List[Document]]:
        docs, texts = [], []
        # download all articles
        for url in urls:
            article = Article(url)
            try:
                article.download()
                article.parse()
            except:
                continue
            title, text = article.title, article.text
            texts.append(text)
            docs.append(Document(title, title))
        embeddings = self.embed(texts)
        # assert len(titles) == len(texts) == embeddings.size()[0],\
        #     f"Titles ({len(titles)}), texts ({len(texts)}) and embeddings ({len(embeddings)}) have different dimensions."
        return embeddings, docs