#%%
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import numpy as np

#%%
text = """
"In the wake of coronavirus pandemic in the country, the authorities has announced to cancel Amarnath Yatra scheduled for 2020. "Based upon the circumstances, Shri Amarnathji Shrine Board decided that it is not advisable to hold and conduct this year’s Shri Amarnathji Yatra and expressed its regret to announce the cancellation of Yatra 2020," the Amarnath Shrine Board said.

"The Board further discussed that pandemic has put the health administration system to its limit. The spike has been particularly very sharp in July. Health Workers and Security Forces are also getting infected and the focus of entire Medical, Civil and Police Administration at the moment is on containing the local transmission of the COVID-19 pandemic. The health concerns are so serious that the strain on the health system, along with the diversion in resources to the Yatra, will be immense," the board explained.

For the devotees, the Amarnath Shrine board will conduct live telecast of the darshan in the morning and aarti in the evening. The traditional rituals will be followed. "Chhadi Mubarak shall be facilitated by the government," the board said.

"The Board is aware of and respects the sentiments of millions of devotees and to keep the religious sentiments alive, the Board shall continue the Live telecast/virtual darshan of the morning and evening Aarti. Further, the traditional rituals shall be carried out as per past practice," it added.

Every year around 10 lakh devotees visit the Amarnath cave shrine. This is the second year in a row that the pilgrimage would not be carried out.

This decision "would enable the Health, Civil and Police Administrations to focus on the immediate challenges facing them rather than diverting resources, manpower and attention to the conduct of the Shri Amarnathji Yatra," the board mentioned.

The Supreme Court had earlier dismissed a petition seeking direction to the Centre, Jammu and Kashmir administration and SASB to cancel this year's Amarnath Yatra.

"The issue as to whether yatra is held should be left with the local administration. As per principles of separation of powers, we leave it to the executive," the apex court said"
"""
#%%
# split text into sentences using nltk
sentences = sent_tokenize(text)
#%%
# Load english stop words
stopwords = set(stopwords.words("english"))
#%%
# Preprocess function. lower case and no stop words
def preprocess(sentence):
    return [
        word.lower()
        for word in word_tokenize(sentence)
        if (word not in stopwords and word.isalpha())
    ]


#%%
# Preprocess all sentences
processed_sentence = list(map(preprocess, sentences))
# Total number of nodes(sentences)
nodes_count = len(list(processed_sentence))

#%%
# create an empty matrix of N x N. N is number of sentences
graph = np.zeros((nodes_count, nodes_count))
for i in range(nodes_count):
    for j in range(i + 1, nodes_count):
        # TextRank computation. Common words in sentences i & j divide by total words in i &j
        graph[i, j] = float(
            len(set(processed_sentence[i]) & set(processed_sentence[j]))
        ) / (len(processed_sentence[i]) + len(processed_sentence[j]))
        graph[j, i] = graph[i, j]

#%%
print(graph)

#%%
# Method to calculate Page Rank (PR) score for each and every sentence.
# PR of sentence x is simply sum of score of all sentences pointing to x.
# Final score is multiplied by a damping factor, generally 0.85
node_weights = np.ones(nodes_count)

#%%
def text_rank_sent(graph, node_weights, d=0.85, iterations=20):
    weight_sum = np.sum(graph, axis=0)
    print(weight_sum)
    while iterations > 0:
        for i in range(len(node_weights)):
            temp = 0.0
            for j in range(len(node_weights)):
                temp += graph[i, j] * node_weights[j] / weight_sum[j]
                print(
                    graph[i, j],
                    node_weights[j],
                    weight_sum[j],
                    graph[i, j] * node_weights[j] / weight_sum[j],
                )
            print(f"temp is {temp}, i is {i}")
            node_weights[i] = 1 - d + (d * temp)
        iterations -= 1


#%%
text_rank_sent(graph, node_weights)

#%%
print(*list(zip(sentences, node_weights)), sep="\n")


# %%
# Print Top N sentences

top_n = 5
top_index = [
    i
    for i, j in sorted(enumerate(node_weights), key=lambda x: x[1], reverse=True)[
        :top_n
    ]
]

top_sentences = [sentences[i] for i in top_index]
print(top_sentences)

# %%
