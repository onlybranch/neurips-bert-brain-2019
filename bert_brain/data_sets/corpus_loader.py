from collections import OrderedDict

from pytorch_pretrained_bert import BertTokenizer

from .spacy_token_meta import make_tokenizer_model
from .corpus_base import CorpusBase


__all__ = ['CorpusLoader']


class CorpusLoader(object):

    def __init__(self, cache_path, bert_pre_trained_model_name='bert-base-uncased'):
        self.bert_pre_trained_model_name = bert_pre_trained_model_name
        self.cache_path = cache_path

    def make_bert_tokenizer(self):
        return BertTokenizer.from_pretrained(self.bert_pre_trained_model_name, self.cache_path, do_lower_case=True)

    def load(
            self,
            index_run,
            corpora,
            dat