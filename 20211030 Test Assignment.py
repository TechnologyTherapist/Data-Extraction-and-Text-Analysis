#!/usr/bin/env python
# coding: utf-8

# ## Import Library

# In[47]:


import plotly.graph_objs
import requests
import pandas as pd
from bs4 import BeautifulSoup
import string
import re
import urllib


# # Text Analysis
# ## Scrap Data

# In[48]:


url="https://insights.blackcoffer.com/how-is-login-logout-time-tracking-for-employees-in-office-done-by-ai/"


# In[49]:


headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}
# Here the user agent is for Edge browser on windows 10. You can find your browser user agent from the above given link.
r = requests.get(url=url, headers=headers)


# In[50]:


soup = BeautifulSoup(r.content, 'html5lib')


# ### Extract title from artical

# In[51]:


title=soup.find('h1',class_="entry-title")
title=title.text.replace('\n'," ")
title


# ### Extract content from articel

# In[52]:


content=soup.findAll(attrs={'class':'td-post-content'})
content=content[0].text.replace('\n'," ")
content


# ### Remove punctuation from the content

# In[53]:


content = content.translate(str.maketrans('', '', string.punctuation))
content


# ### Convert Tokens

# In[54]:


from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(content)
print(text_tokens[0:50])


# ### Remove stopwords from the tokens

# In[55]:


import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

my_stop_words = stopwords.words('english')
my_stop_words.append('the')
no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens[0:40])


# ### Check for positive words

# In[56]:


with open("positive-words.txt",'r') as pos_word:
     pos_words = pos_word.read().split("\n")
     pos_words = pos_words[5:]


# ### Positive Score

# In[57]:


pos_count = " ".join([w for w in no_stop_tokens if w in pos_words])
pos_count = pos_count.split(" ")
Positive_score=len(pos_count)
print(Positive_score)


# ### Check for negative words

# In[58]:


with open("negative-words.txt","r") as neg:
   negwords = neg.read().split("\n")

negwords = negwords[36:]
neg_count = " ".join ([w for w in no_stop_tokens if w in negwords])
neg_count=neg_count.split(" ")


# ### NEgative Score

# In[59]:


Negative_score=len(neg_count)
print(Negative_score)


# In[60]:


filter_content = ' '.join(no_stop_tokens)


# In[61]:


data=[[url,title,content,filter_content,Positive_score,Negative_score]]


# In[62]:


data=pd.DataFrame(data,columns=["url","title","content","filter_content","Positive_Score","Negative_Score"])


# ### Calculate Polarity Score and Subjectivity Score

# In[63]:


from textblob import TextBlob
def sentiment_analysis(data):
   sentiment=TextBlob(data["content"]).sentiment
   return pd.Series([sentiment.polarity,sentiment.subjectivity])
data[["polarity","subjectivity"]]=data.apply(sentiment_analysis,axis=1)
data


# # Average Sentence length

# In[64]:


AVG_SENTENCE_LENGTH = len(content.replace(' ',''))/len(re.split(r'[?!.]', content))
print('Word average =', AVG_SENTENCE_LENGTH)


# ## FOG Index

# In[65]:


import textstat
FOG_index=(textstat.gunning_fog(content))
print(FOG_index)


# ## AVG Number of words per Sentence

# In[66]:


AVG_NUMBER_OF_WORDS_PER_SENTENCE = [len(l.split()) for l in re.split(r'[?!.]', content) if l.strip()]
AVG_NUMBER_OF_WORDS_PER_SENTENCE=print(sum(AVG_NUMBER_OF_WORDS_PER_SENTENCE)/len(AVG_NUMBER_OF_WORDS_PER_SENTENCE))


# ## Complex word Count

# In[67]:


def syllable_count(word):
    count = 0
    vowels = "AEIOUYaeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("es"or "ed"):
                count -= 1
    if count == 0:
        count += 1
    return count



COMPLEX_WORDS=syllable_count(content)
print(COMPLEX_WORDS)


# # Word Counts

# In[68]:


Words_counts=len(content)
print(Words_counts)


# # Percentage of Complex words

# In[69]:


pcw=(COMPLEX_WORDS/Words_counts)*100
print(pcw)


# ## Personal Pronouns

# In[70]:


def ProperNounExtractor(text):
    count = 0
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        for (word, tag) in tagged:
            if tag == 'PRP': # If the word is a proper noun
                count+=1

    return(count)



# Calling the ProperNounExtractor function to extract all the proper nouns from the given text.
Personal_Pronouns=ProperNounExtractor(content)


# # Average Word Length

# In[71]:


Average_Word_Length=len(content.replace(' ',''))/len(content.split())
print(Average_Word_Length)


# 
# # SYLLABLE PER WORD

# In[72]:


word=content.replace(' ','')
syllable_count=0
for w in word:
      if(w=='a' or w=='e' or w=='i' or w=='o' or w=='y' or w=='u' or w=='A' or w=='E' or w=='I' or w=='O' or w=='U' or w=='Y'):
            syllable_count=syllable_count+1
print("The AVG number of syllables in the word is: ")
print(syllable_count/len(content.split()))


# ### For WORDCLOUD

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud)
    # No axis details
    plt.axis("off")
stopwords = STOPWORDS
stopwords.add('will')
wordcloud = WordCloud(width = 500, height = 500, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(content)
# Plot
plot_cloud(wordcloud)


# # Negative wordcloud

# In[ ]:


neg_review = " ".join ([w for w in neg_count if w in negwords])

wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(neg_review)
#Plot
plot_cloud(wordcloud)

