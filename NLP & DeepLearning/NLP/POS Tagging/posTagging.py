import nltk
import spacy

text = ["Time flies like an arrow", "small boys and girls"]

# code starts here
pos_tags = []
for t in text:
    token = nltk.word_tokenize(t)
    post = nltk.pos_tag(token)
    pos_tags.append(post)

print(pos_tags)


nlp = spacy.load("en_core_web_sm")
doc = nlp("Time flies like an arrow")
print(type(doc))
pos_tags = {}
for token in doc:
    print(token)
    pos_tags[token.text] = token.pos_
print(pos_tags)
