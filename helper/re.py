import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def get_title(title:str, check:list=['Mr','Miss','Mrs','Master']):
    """타이틀에서 알파벳을 제외한 나머지 글자를 제외하고 어절단위로 리스트로 묶은 후 체크해야할 리스트에 있는 단어가 존재시 해당 단어를 리턴하고 그외에는 Rare를 리턴함, 만약 str이 아닐경우 NOT STR을 리턴

    Args:
        name (str): 타이틀
        check (list, optional) : 확인할 리스트.
    
    """
    try:
        title_search = re.sub('[^A-Za-z]',' ',title)
        
        for i in (title_search.split()):
            if i in check:
                return i
        else:
             return 'Rare'   
    except:
        return 'NOT_STR'
    
def remove_emails(x):
    email_pattern = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
    return re.sub(email_pattern, '', x)

def remove_HTML(x):
    return BeautifulSoup(x).get_text().strip()

def remove_special_chars(x):
    x = re.sub(r'[^\w ]+', "", x)
    x = ' '.join(x.split())
    return x

nltk.download('stopwords')
sw = stopwords.words('english')

def remove_stopwords(x):
    global sw
    if type(x) == list:
        return ' '.join([word for word in x if word not in sw])
    elif type(x) == str:
        return  ' '.join([word for word in x.split() if word not in sw])
    
def get_stem(x):
    return [PorterStemmer.stem(i) for i in x]

