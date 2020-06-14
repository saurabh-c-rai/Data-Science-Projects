#%%
glove_path = "../models/glove.6B.100d.txt.word2vec"

#%%
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# %%
model = KeyedVectors.load_word2vec_format(fname=glove_path, binary=False)

# %%
result = model.most_similar(positive=["rome", "france"], negative=["italy"], topn=1)

# %%
model.most_similar(positive=["woman", "father"], negative=["man"], topn=1)


# %%
