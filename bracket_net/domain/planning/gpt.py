from ...model import gpt

from .util import NaiveBase, NNAstarLikeBase


class Naive(NaiveBase):
    def __init__(self, config):
        super().__init__(config, gpt.GPT)

 
class NNAstarLike(NNAstarLikeBase):
    def __init__(self, config):
        super().__init__(config, gpt.GPT)


if __name__ == '__main__':
    model = Naive()
    model = NNAstarLike()