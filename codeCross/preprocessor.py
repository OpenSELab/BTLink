import Stemmer
from nltk.tokenize import WordPunctTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import nltk.data
import re

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer('english')


def RemoveHttp(str):
    httpPattern = '[a-zA-z]+://[^\s]*'
    return re.sub(httpPattern, ' ', str)


def RemoveTag(str, key):
    keys = key.split("/")
    patterns = [k + '-[0-9]*' for k in keys]
    pattern = ""
    for ip in range(len(patterns)):
        if ip == 0:
            pattern = patterns[ip]
        else:
            pattern = pattern + "|" + patterns[ip]
    return re.sub(pattern, ' ', str)


def clean_en_text(text):
    # keep English, digital and space
    comp = re.compile('[^A-Z^a-z^0-9^ ]')
    return comp.sub(' ', text)


def RemoveGit(str):
    gitPattern = '[Gg]it-svn-id'
    return re.sub(gitPattern, ' ', str)


def textProcess(text, key):
    final = []
    text = RemoveHttp(text)
    text = RemoveTag(text, key)
    text = RemoveGit(text)
    sentences = tokenizer.tokenize(text)
    for sentence in sentences:
        sentence = clean_en_text(sentence)
        word_tokens = word_tokenize(sentence)
        word_tokens = [word for word in word_tokens if word.lower() not in stopwords.words('english')]
        for word in word_tokens:
            if word in stopwords.words('english'):
                continue
            else:
                final.append(str(stemmer.stem(word)))
    if len(final) == 0:
        text = ' '
    else:
        text = ' '.join(final)
    return text


def codeMatch(word):
    identifier_pattern = r'''[A-Zz-z]+[0-9]*_.*
                            |[A-Za-z]+[0-9]*[\.].+
                            |[A-Za-z]+.*[A-Z]+.*
                            |[A-Z0-9]+
                            |_ +[A-Za-z0-9]+.+
                            |[a-zA-Z]+[:]{2,}.+   
                            '''
    identifier_pattern = re.compile(identifier_pattern)
    if identifier_pattern.match(word):
        return True
    else:
        return False


def diffProcess(text):
    identifiers = []
    sentences = tokenizer.tokenize(text)
    for sentence in sentences:
        word_tokens = word_tokenize(sentence)
        for word in word_tokens:
            if codeMatch(word):
                identifiers.append(word)
    if len(identifiers) == 0:
        text = ' '
    else:
        text = ' '.join(identifiers)
    return text
