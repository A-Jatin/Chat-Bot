import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_colwidth',200)
df=pd.read_csv('a.csv')
df
convo = df.iloc[:,0]
convo
clist = []

def qa_pairs(x):
  cpairs = re.findall(": (.*?)(?:$|\n)", x)
  clist.extend(list(zip(cpairs, cpairs[1:])))
convo.map(qa_pairs);
convo_frame = pd.Series(dict(clist)).to_frame().reset_index()
convo_frame.columns = ['q', 'a']

vectorizer = TfidfVectorizer(ngram_range=(1,3))
vec = vectorizer.fit_transform(convo_frame['q'])

def get_response(q):
  my_q = vectorizer.transform([q])
  
  cs = cosine_similarity(my_q, vec)
  
  rs = pd.Series(cs[0]).sort_values(ascending=False)
  
  rsi = rs.index[0]
  return convo_frame.iloc[rsi]['a']
print("Hey!")

while(1==1):
    
    ans=input()
    if(ans=="exit" or ans== "Exit"):
        break
    print("Bot : " + get_response(ans))
    
