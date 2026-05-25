# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
# nltk.download('stopwords')

paragraph = '''Artificial Intelligence refers to the intelligence of machines. This is in contrast to the natural intelligence of
humans and animals. With Artificial Intelligence, machines perform functions such as learning, planning, reasoning and
problem-solving. Most noteworthy, Artificial Intelligence is the simulation of human intelligence by machines.
It is probably the fastest-growing development in the World of technology and innovation. Furthermore, many experts believe
AI could solve major challenges and crisis situations.'''

sentences = nltk.sent_tokenize(paragraph)
sentences

stemmer = PorterStemmer()

for i in range(len(sentences)):
    words =nltk.word_tokenize(sentences[i])
    words=[stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i]= ' '.join(words)
    
sentences