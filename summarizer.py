# need to install nptyping and use nltk.download('punkt')
import numpy as np
from typing import AnyStr, Callable, List
from nptyping import NDArray, Float64
import pandas as pd 
import numpy as np 
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import string

class WordEmbedding:
  '''
  Attributes
  -----------------------------
  embedding_size: int
    it's the embedding size of the word.

  Methods
  -----------------------------
  get_word_embedding(word: str) -> ndarray[float64]
    gives the embedding representation of a word

  get_sent_embedding(sent: List[str]) -> ndarray[float64]
    gives the embedding representation of a sentence, by computing
    the mean of the word embedding of its words.
  '''
  def __init__(self,filename:AnyStr):
    '''
    parameters
    -----------------------------
    filename: str
      represent the file that contains the embeddings,
      it assume the file has the following structure: 
      
      word1 .23 .45 .56
      word2 .23 .45 .56
      word3 .23 .45 .56
    '''
    self.filename = filename
    self.__init_embedd_size()
    self.__init_dict()
  
  def get_word_embedding(self, word: AnyStr) -> NDArray[Float64]:
    '''
    parameters
    -----------------------------
    word: str
      a word 

    returns
    -----------------------------
    embedding: ndarray 
      a vector embedding for the word
    '''
    embedding = np.zeros(self.embedding_size, dtype=np.float)
    if word in self.__word_embedding:
      embedding = self.__word_embedding[word]
    return embedding
  
  def get_sent_embedding(self, sent: List[str]) -> NDArray[Float64]:
    '''
    parameters
    -----------------------------
    sent: list[AnyStr]
      an array of words

    returns
    -----------------------------
    sent_rep: NDArray[Float64]
      a vector embedding representing the sentence.
    '''
    sent_rep = np.zeros(self.embedding_size)
    num_of_valid_words = 0
    for word in sent: 
      if word in self.__word_embedding:
        sent_rep += self.get_word_embedding(word)
        num_of_valid_words+=1
    
    sent_rep = sent_rep / num_of_valid_words
    return sent_rep


  def __init_embedd_size(self):
    '''init embedding size'''
    f = open(self.filename)
    line = next(iter(f))
    self.embedding_size = len(line.split(' ')) - 1
    f.close()

  def __init_dict(self):
    '''init word embedding dictionary from a file'''
    self.__word_embedding = {}
    word_embedding = {}
    f = open(self.filename)
    for line in f:
      line = line.split(' ')
      vector = [float(x) for x in line[1:]]
      word = line[0]

      self.__word_embedding[word] = np.array(vector)
    f.close()
    


class Summarizer:
  '''
  A summarizer that uses KMeans algorithm. It's an extractive summarizer.
  
  Methods
  -----------------------------
  summarize(text: str) -> str
    summarize given text
  '''
  def __init__(self, word_embedding: WordEmbedding, preprocess_text: Callable,\
               sent_tokenizer: Callable, word_tokenizer:Callable):
    '''
    parameters
    -----------------------------
    word_embedding_file: str
      the path to the embedding file which contains word embedding
    
    preprocess_text: Callable str -> str
      function to preprocess the text

    sent_tokenizer: Callable str -> list[str]
      tokenize text into sentences
    
    word_tokenizer: Callable str -> list[str]
      tokenize text into words
    '''
    self.word_embedding = word_embedding
    self.preprocess_text = preprocess_text
    self.sent_tokenizer = sent_tokenizer
    self.word_tokenizer = word_tokenizer
    

  def summarize(self, text: AnyStr) -> AnyStr:
    sentences = self.sent_tokenizer(text)
    for i in range(len(sentences)):
      sentences[i] = self.preprocess_text(sentences[i])
    number_of_sent = len(sentences)

    X = np.zeros((len(sentences), self.word_embedding.embedding_size))
    for i,sent in enumerate(sentences):
      tokens = self.word_tokenizer(sent)
      sent_rep = self.word_embedding.get_sent_embedding(tokens)
      X[i] = sent_rep
    
    num_clusters = int(number_of_sent ** 0.5) + 1
    model = KMeans(n_clusters=num_clusters)


    model.fit(X)

    average_order_of_cluster = [0] * num_clusters

    for k in range(num_clusters):
      idx = np.where(model.labels_ == k)[0]
      idx_avg = np.mean(idx)
      average_order_of_cluster[k] = idx_avg

    closest, _ = pairwise_distances_argmin_min(model.cluster_centers_, X)
    orderings = range(num_clusters) 
  
    orderings = sorted(orderings, key= lambda x: average_order_of_cluster[x])
    summary = '.'.join([sentences[closest[j]] for j in orderings])
    
    return summary
    

def preprocess_sent(sent):
  regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
  sent = regrex_pattern.sub(r'',sent)


  sent = sent.lower()

  tokens = word_tokenize(sent)


  p_tokens = []
  for token in tokens:
    if token in string.punctuation:
      continue
    else:
      p_tokens.append(token)
  return ' '.join(p_tokens)

