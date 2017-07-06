import os
import numpy as np
np.random.seed(1337)
import sys
import nltk
from nltk import *
from nltk.corpus import *
#from nltk.stem import WordNetLemmatizer
#wordnet_lemmatizer = WordNetLemmatizer()
from scipy import stats
from scipy import spatial
#import wordsegment
#from wordsegment import *
#from wordsegment import segment

def word_embeddings():
  import os
  import numpy as np
  np.random.seed(1337)
  import sys
  import nltk
  from nltk.stem import WordNetLemmatizer
  wordnet_lemmatizer = WordNetLemmatizer()
  from scipy import stats
  from scipy import spatial
  BASE_DIR = '/home/buildadmin/'
  GLOVE_DIR = BASE_DIR + '/glove.6B/'
  MAX_SEQUENCE_LENGTH = 1000
  MAX_NB_WORDS = 20000
  EMBEDDING_DIM = 100
  VALIDATION_SPLIT = 0.2
  word_list3=[]
  embeddings_index3 = {}
  f = open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'))
  for line in f:
      values = line.split()
      word = values[0]
      if word not in set(stopwords.words('english')):
          if word.isalnum():
                word_list3.append(values[0])
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index3[word] = coefs
  f.close()
  return embeddings_index3


def semantic_mean3(q1,q2):
  q2 = re.sub('[^a-zA-Z \n\.]', '', q2)
  q2=q2.lower()
  q2=q2.split()
  w=[]
  for w1 in q2:
      if w1 not in stopwords.words('english'):
          w.append(w1)
  q2=w
  q1 = re.sub('[^a-zA-Z \n\.]', '', q1)
  q1=q1.lower()
  q1=q1.split()
  w=[]
  for w1 in q1:
      if w1 not in stopwords.words('english'):
          w.append(w1)
  q1=w
  score=[]
  vec1=[]
  vec2=[]
  q1=np.unique(q1)
  q2=np.unique(q2)
  for w1 in q1:
    for w2 in q2:
      vec1=embeddings_index3.get(w1)
      vec2=embeddings_index3.get(w2)
      try:
        score.append(1-spatial.distance.cosine(vec1, vec2))
      except Exception:
        pass
  if (len(score)!=0):
    score=np.mean(score)
  else:
    score=0
  return score




def pos_clean(q):
  w=[]
  nn=[]
  nns=[]
  jj=[]
  vbn=[]
  nn=np.array(nltk.pos_tag(nltk.word_tokenize(q)))[np.array(nltk.pos_tag(nltk.word_tokenize(q)))[np.arange(len(np.array(nltk.pos_tag(nltk.word_tokenize(q))))),1]=='NN']
  nns=np.array(nltk.pos_tag(nltk.word_tokenize(q)))[np.array(nltk.pos_tag(nltk.word_tokenize(q)))[np.arange(len(np.array(nltk.pos_tag(nltk.word_tokenize(q))))),1]=='NNS']
  vbn=np.array(nltk.pos_tag(nltk.word_tokenize(q)))[np.array(nltk.pos_tag(nltk.word_tokenize(q)))[np.arange(len(np.array(nltk.pos_tag(nltk.word_tokenize(q))))),1]=='VBN']
  vbg=np.array(nltk.pos_tag(nltk.word_tokenize(q)))[np.array(nltk.pos_tag(nltk.word_tokenize(q)))[np.arange(len(np.array(nltk.pos_tag(nltk.word_tokenize(q))))),1]=='VBG']
  jj=np.array(nltk.pos_tag(nltk.word_tokenize(q)))[np.array(nltk.pos_tag(nltk.word_tokenize(q)))[np.arange(len(np.array(nltk.pos_tag(nltk.word_tokenize(q))))),1]=='JJ']
  nn=np.array(np.unique(nn[np.arange(len(nn)),0]))
  nns=np.array(np.unique(nns[np.arange(len(nns)),0]))
  jj=np.array(np.unique(jj[np.arange(len(jj)),0]))
  vbn=np.array(np.unique(vbn[np.arange(len(vbn)),0]))
  vbg=np.array(np.unique(vbg[np.arange(len(vbg)),0]))
  list=[]
  list.extend(nn)
  list.extend(nns)
  list.extend(jj)
  list.extend(vbn)
  list.extend(vbg)
  list=" ".join(list)
  return list


def exact_pos_sim(q1,q2):
  list1=[]
  list2=[]
  a_mean=[]
  list1=nltk.pos_tag(word_tokenize(q1))
  list1=np.array(list1)
  list2=nltk.pos_tag(word_tokenize(q1))
  list2=np.array(list1)
  a_temp=[]
  for i in np.arange(len(list1)):
    for j in np.arange(len(list2)):  
       if ('VB' in list1[i,1]):
         if ('VB' in list2[i,1]):
           vec1=embeddings_index3.get(list1[i,0])
           vec2=embeddings_index3.get(list2[j,0])
           try:
              a_temp.append(1-spatial.distance.cosine(vec1,vec2))
           except:
              pass
  a_mean.append(np.mean(a_temp))
  a_temp=[]
  for i in np.arange(len(list1)):
    for j in np.arange(len(list2)):  
       if ('JJ' in list1[i,1]):
         if ('JJ' in list2[i,1]):
           vec1=embeddings_index3.get(list1[i,0])
           vec2=embeddings_index3.get(list2[j,0])
           try:
              a_temp.append(1-spatial.distance.cosine(vec1,vec2))
           except:
              pass
  a_mean.append(np.mean(a_temp))
  a_temp=[]
  for i in np.arange(len(list1)):
    for j in np.arange(len(list2)):  
       if ('NN' in list1[i,1]):
         if ('NN' in list2[i,1]):
           vec1=embeddings_index3.get(list1[i,0])
           vec2=embeddings_index3.get(list2[j,0])
           try:
              a_temp.append(2*(1-spatial.distance.cosine(vec1,vec2)))
           except:
              pass
  a_mean.append(np.mean(a_temp))
  a_temp=[]
  for i in np.arange(len(list1)):
    for j in np.arange(len(list2)):  
       if ('RB' in list1[i,1]):
         if ('RB' in list2[i,1]):
           vec1=embeddings_index3.get(list1[i,0])
           vec2=embeddings_index3.get(list2[j,0])
           try:
              a_temp.append(1-spatial.distance.cosine(vec1,vec2))
           except:
              pass
  a_mean.append(np.mean(a_temp))
  a_temp=[]
  a_mean=np.array(a_mean)
  for i in np.arange(len(a_mean)):
    if (np.isnan(a_mean[i])==False):
      a_temp.append(a_mean[i])
  return np.mean(a_temp)



def chatbot1(q):
  q_correct=semantic_correct(q)[0]
  q1=q_correct
  w=[]  
  for w1 in q_correct.lower().split(): 
    if w1 not in stopwords.words('english'):
       w.append(w1)
  q_correct=' '.join(w)
  a1=[]
  a=[]
  ques=exact_match(q_correct)
  for i in np.arange(len(x)):
     a1.append(exact_pos_sim(x[i,3],q_correct))
  s1=[]
  for s in x[(np.argsort(a1)[::-1][np.arange(7)]),0]:
    s1.append(s)
  a2=[]
  for i in np.arange(len(ques)):
     a2.append(exact_pos_sim(x[i,3],q_correct))
  s2=[]
  if (len(ques)>0):
    if (len(ques)<3):
      for s in ques[(np.argsort(a2)[::-1]),0]:
          s2.append(s)
    else:
      for s in ques[(np.argsort(a2)[::-1][np.arange(3)]),0]:
          s2.append(s)
  if (int(semantic_correct(q)[1])==1):
      res=[]
      t='Did You Mean?:: ' + q1
      res.append(t)
      for i in np.arange(len(s2)):
        res.append(s2[i])
      for i in np.arange(len(s1)):
        res.append(s1[i])
        if (len(a2)>0):
          a.append(np.max(a2))
        else:
          a.append(0)
        if (len(a1)>0):
          a.append(np.max(a1))
        else:
          a.append(0)
      if (np.mean(a)<0.25):
        res='Sorry, I could not find and answer to your question, but I will learn an answer to that soon!!'
  else:
      res=[]
      for i in np.arange(len(s2)):
        res.append(s2[i])
      for i in np.arange(len(s1)):
        res.append(s1[i])
        if (len(a2)>0):
          a.append(np.max(a2))
        else:
          a.append(0)
        if (len(a1)>0):
          a.append(np.max(a1))
        else:
          a.append(0)
      if (np.mean(a)<0.25):
        res='Sorry, I could not find and answer to your question, but I will learn an answer to that soon!!'
  return np.array(np.unique(res))



def semantic_correct(q):
  flag=0
  q=q.replace('blood pressure','bp')
  q1=np.array(q.lower().split())
  import enchant
  correct=[]
  wrong=[]
  d=enchant.Dict("en_US")
  for i in np.arange(len(q1)):
     if (q1[i] in exceptions):
         correct.append(q1[i])
     else:
       if (not d.check(q1[i])):
          wrong.append(q1[i])
       else:
          correct.append(q1[i])
  if (len(wrong)>0):
    wrong_correct=[]
    for i in np.arange(len(wrong)):
        suggest_sem_score=[]
        suggest=d.suggest(wrong[i])  
        for j in np.arange(len(suggest)):
             suggest_score=np.arange(1,len(suggest)+1)[::-1]
             suggest_sem=[]
             for k in np.arange(len(correct)):
                 vec1=embeddings_index3.get(suggest[j])
                 vec2=embeddings_index3.get(correct[k])
                 try:
                    suggest_sem.append(1-spatial.distance.cosine(vec1,vec2))
                 except:
                    suggest_sem.append(0)
             suggest_sem_score.append(np.mean(suggest_sem)*suggest_score[j])           
        wrong_correct.append(suggest[np.argmax(suggest_sem_score)])
    sent=[]      
    for i in q1:
        try:
           sent.append(correct[correct.index(i)])
        except:
           sent.append(wrong_correct[wrong.index(i)])
    flag=1    
  else:
    sent=q1     
  return np.array([' '.join(sent),flag])


def exact_match(q1):
  n1=[]
  q_list=[]
  w=[]
  q1=pos_clean(q1)
  for w1 in q1.lower().split(): 
    if w1 not in stopwords.words('english'):
       w.append(w1)
  q1=" ".join(w)    
  for i in np.arange(len(x)):
    if (len(set(q1.lower().split()).intersection(set(x[i,3].lower().split())))>0):
      n1.append(len(set(q1.lower().split()).intersection(set(x[i,3].lower().split()))))
      q_list.append(x[i,0])
  if (len(n1)>0):
    q_list=np.array(q_list)[((np.argsort(n1)[::-1]))]
      #[np.arange(((len(n1)+2)/2))]
    return np.array(q_list).reshape(len(q_list),1)
  else:
    return np.array([])


exceptions=['bp','cellulitis','hypotension','ayurveda','ayurvedic','paneer','creatine','aloe','vera','amla','roti','fbs','rbs','pcos','neem','tv']


def chatbot(q):
  q_correct=semantic_correct(q)[0]
  q1=q_correct
  w=[]  
  for w1 in q_correct.lower().split(): 
    if w1 not in stopwords.words('english'):
       w.append(w1)
  q_correct=' '.join(w)
  q_correct=pos_clean(q_correct)
  a1=[]
  a=[]
  ques=exact_match(q_correct)
  for i in np.arange(len(x)):
     a1.append(semantic_mean3(x[i,2],q_correct)+semantic_mean3(x[i,3],q_correct))
     #a1.append(semantic_mean3(x[i,3],q_correct))
  s1=[]
  for s in x[(np.argsort(a1)[::-1][np.arange(7)]),0]:
    s1.append(s)
  a2=[]
  for i in np.arange(len(ques)):
     a2.append(semantic_mean3(x[i,2],q_correct)+semantic_mean3(x[i,3],q_correct))
     #a2.append(semantic_mean3(ques[i,0],q_correct))
  s2=[]
  if (len(ques)>0):
    if (len(ques)<3):
      for s in ques[(np.argsort(a2)[::-1]),0]:
          s2.append(s)
    else:
      for s in ques[(np.argsort(a2)[::-1][np.arange(3)]),0]:
          s2.append(s)
  if (int(semantic_correct(q)[1])==1):
      res=[]
      t='Did You Mean?:: ' + q1
      res.append(t)
      for i in np.arange(len(s2)):
        res.append(s2[i])
      for i in np.arange(len(s1)):
        res.append(s1[i])
        if (len(a2)>0):
          a.append(np.max(a2))
        else:
          a.append(0)
        if (len(a1)>0):
          a.append(np.max(a1))
        else:
          a.append(0)
      if (np.mean(a)<0.25):
        res='Sorry, I could not find and answer to your question, but I will learn an answer to that soon!!'
  else:
      res=[]
      for i in np.arange(len(s2)):
        res.append(s2[i])
      for i in np.arange(len(s1)):
        res.append(s1[i])
        if (len(a2)>0):
          a.append(np.max(a2))
        else:
          a.append(0)
        if (len(a1)>0):
          a.append(np.max(a1))
        else:
          a.append(0)
      if (np.mean(a)<0.25):
        res='Sorry, I could not find and answer to your question, but I will learn an answer to that soon!!'
  return np.array(np.unique(res))
 

##################################### end of all functions
#import csv
#from random import randint
#reader=csv.reader(open("chronic qna.csv","rb"),delimiter=',')
#x=list(reader)
#x=np.array(x)
#for i in np.arange(len(x)):
#   x[i,0]=(" ".join(x[i,0].split("\n")))
#   x[i,1]=(" ".join(x[i,1].split("\n")))
#   x[i,3]=pos_clean(x[i,0])
#   x[i,4]=pos_clean(x[i,1])

import csv
from random import randint
reader=csv.reader(open("chronic qna cleaned.csv","rb"),delimiter=',')
x=list(reader)
x=np.array(x)
x=x[1:]


#embeddings_index3=word_embeddings()
embeddings_index3 = np.load('embeddings_index3.npy').item()

from Tkinter import *
root = Tk()
l=Label(root,text="Retrival Agent 1.0")
l.pack()
root.geometry("500x500")
label1 = Label( root, text="Question")
E1 = Entry(root, bd =5, width=50)
t = Text(root,font=("Helvetica", 10))


def getans():
    q=E1.get()
    ans=chatbot(q)
    t.delete("1.0",END)
    display(ans)
    

def display(ans):
    for x in ans:
        t.insert(END, x + '\n')
    

submit = Button(root, text ="Submit", command = getans)
t.pack()
label1.pack()
E1.pack()
submit.pack(side =BOTTOM) 
root.mainloop()


