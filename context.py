from typing import List, Dict
from topic import Topic
from searcher import Searcher
from vector_space import VectorSpace

class Context(VectorSpace):

    def __init__(self):
        super(Context, self).__init__()
        self._topics = []
        self._searcher = Searcher()

    def optimize(self):
        pass

    def addTopic(self, new_topic: Dict[str, Topic]):
        pass

    def removeTopic(self):
        pass