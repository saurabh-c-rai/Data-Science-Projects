# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
#%%
# working directory
import os

cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)

# %%
# Import NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# %%
text = "Avengers: Endgame is an upcoming American superhero film based on the Marvel Comics superhero team the Avengers, produced by Marvel Studios and set for distribution by Walt Disney Studios Motion Pictures. It is the direct sequel to 2018's Avengers: Infinity War, a sequel to 2012's Marvel's The Avengers and 2015's Avengers: Age of Ultron, and the 22nd film in the Marvel Cinematic Universe (MCU). The film is directed by Anthony and Joe Russo with a screenplay by Christopher Markus and Stephen McFeely and features an ensemble cast including Robert Downey Jr., Chris Evans, Mark Ruffalo, Chris Hemsworth, Scarlett Johansson, Jeremy Renner, Don Cheadle, Paul Rudd, Brie Larson, Karen Gillan, Danai Gurira, Bradley Cooper, and Josh Brolin. In the film, the surviving members of the Avengers and their allies work to reverse the damage caused by Thanos in Infinity War.The film was announced in October 2014 as Avengers: Infinity War – Part 2. The Russo brothers came on board to direct in April 2015, and by May, Markus and McFeely signed on to script the film. In July 2016, Marvel removed the title, referring to it simply as Untitled Avengers film. Filming began in August 2017 at Pinewood Atlanta Studios in Fayette County, Georgia, shooting back-to-back with Infinity War, and ended in January 2018. Additional filming took place in the Metro and Downtown Atlanta areas and New York. The official title was revealed in December 2018."


# %%
token = word_tokenize(text)


# %%
tag = pos_tag(token)


# %%
ne_chunks = nltk.ne_chunk(tag)


# %%
for name in ne_chunks:
    if hasattr(name, "label"):
        print(name.label(), " - ".join(c[0] for c in name.leaves()))


# %%
import spacy


# %%
model = spacy.load("en_core_web_sm")


# %%
parsed = model(text)


# %%
for entity in parsed.ents:
    print(entity.text, entity.label_)


# %%

