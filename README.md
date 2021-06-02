# text-summarizer
It is a summarizer python code based on kmeans clustering to give an extractive summarization. It's very fast and it gives very satisfying results.

## How to use it
The project contains two classes in summarizer.py I will explain both of them below:
### WordEmbedding
This class is responsible for providing word embedding for words and sentences. It uses a pre-trained word embedding file and load into a dictionary. This pre-trained 
word embedding file should has the following format:

word1 0.12 0.34 ... 0.67\
word2 0.45 0.67 ... 0.78\
     ............\
wordn 0.21 0.11  ... 0.56

#### Note !!
To download a pre-trained word embedding file you can visit stanford site to download glov word embedding
here is the link [link](https://nlp.stanford.edu/projects/glove/)

### Summarizer 
This class is responsible for summarization. This class expects to get the following when intializing it:
- WordEmbedding ( and I explained above ). 
- preprocess_text which is a function that preprocess the text like converting it to lower case and may be remove punctuations. 
- sent_tokenizer which is a function that tokenize the text into sentences.
- word_tokenizer which is a function that tokenize the text into words.
to summarize just create an object of it and call its summarize method passing to the text you want to summarize.


