from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk
import gensim
from gensim import corpora
import re

file2Open = open("paragraph.txt","r")
stext = file2Open.read()
doc_complete = nltk.sent_tokenize(stext)

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

# Create a dictionary of terms within the corpus, where every unique term is assigned an index value.
dictionary = corpora.Dictionary(doc_clean)
# Convert list of corpus into Document Term Matrix from dictionary defined above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
# Create the object for LDA model
Lda = gensim.models.ldamodel.LdaModel
# Run and Train the LDA model on the matrix from above.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=3, num_words=3))




strOut = ldamodel.print_topics(num_topics=3, num_words=3)
#strOut = [(0, '0.209*"fruit" + 0.125*"drink" + 0.125*"juice" '),(1, '0.150*"fruit" + 0.149*"eat" + 0.149*"remove" ')]
#print (type(strOut))
#print (strOut[0][-1])
listOfTags = []
for item in strOut:
    #print (item[-1])
    stringTest = item[-1]
    res = re.findall(r'\w+', stringTest)
    tagList = []
    #print (res)
    for item2 in res:
        if (item2.isdigit()==True):
            pass
        else:
            tagList.append(item2)
    listOfTags.append(tagList)
print (listOfTags)
