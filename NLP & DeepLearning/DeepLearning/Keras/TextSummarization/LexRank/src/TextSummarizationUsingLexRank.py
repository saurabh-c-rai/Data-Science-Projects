#%%
text = 'Stake of Airtel promoter group firm Bharti Telecom has come down to 41.24 per cent following renouncement of 11.34 crore \
shares by the entity valued at around Rs 3,920 crore and Rs 25,000-crore rights issue, according to the updated shareholding of \
the company. Sunil Bharti Mittal family and Singapore telecom firm Singtel-owned firm Bharti Telecom held 50.1 per cent stake \
with 211.62 shares in Bharti Airtel at the end of March 2019. Bharti Telecom shareholding dropped to Rs 200.28 crore in May \
disclosure by Bharti Airtel. According to sources, Bharti Telecom renounced shares for the Singapore government and the Monetary \
Authority of Singapore and foreign-based promoter group firm Indian Continent Investment Limited (ICIL). GIC Pvt Ltd, on behalf \
of the Government of Singapore and the Monetary Authority of Singapore, has made a commitment of Rs 5,000 crore during the rights \
issue programme. At the time of the Rs 25,000-crore rights issue, the company allocated shares for Rs 220 a unit. The Singapore \
government entities now jointly hold over 24.73 crore shares. Based on stock closing price of Rs 345.65 a unit on the BSE on \
Tuesday, the value of shares allocated to Singapore government entities now stands at around Rs 8,550 crore and accounts for \
4.82 per cent stake in the Indian telecom firm. ICIL holding in Airtel increased 25 crore shares, valued at around Rs 8,651 \
crore based on stock closing price, to 33.14 crore units in May from 8.11 crore at the end of March 2019. Responding to an \
e-mail query, Bharti Airtel said, "As stated in our letter of offer, the promoter group amongst themselves, have subscribed \
to their aggregate entitlement except to the extent of 227,272,727 shares renounced in favour of GIC Private Limited. \
Bharti Telecom Limited and Indian Continent Investment are part of the promoter group." Airtel had announced the rights \
issue to raise up to Rs 25,000 crore through issuance of fully paid-up shares at a price of Rs 220 per share, and additional \
Rs 7,000 crore through a foreign currency perpetual bond issue. It opened on May 3 and closed on May 17, 2019. While the offer \
price was fixed at Rs 220 per share, the Bharti Airtel scrip ended at Rs 328.20 apiece on the BSE, 0.84 per cent higher than \
the previous close, on the day the rights issue closed.'

# %%
from lexrank import STOPWORDS, LexRank
from nltk.tokenize import sent_tokenize, word_tokenize
from path import Path

#%%
documents = []
documents_dir = Path("../input/bbc/politics")
#%%
for file_path in documents_dir.files("*.txt"):
    with file_path.open(mode="rt", encoding="utf-8") as fp:
        documents.append(fp.readlines())

# %%
lxr = LexRank(documents, stopwords=STOPWORDS["en"])

#%%
# get summary with classical LexRank algorithm
sentences = sent_tokenize(text)
summary = lxr.get_summary(sentences, summary_size=2, threshold=0.1)
print(summary)

#%%
# get summary with continuous LexRank
summary_cont = lxr.get_summary(sentences, threshold=None)
print(summary_cont)

# ['The BBC understands that as chancellor, Mr Osborne, along with the Treasury '
#  'will retain responsibility for overseeing banks and financial regulation.']
#%%
# get LexRank scores for sentences
# 'fast_power_method' speeds up the calculation, but requires more RAM
scores_cont = lxr.rank_sentences(sentences, threshold=None, fast_power_method=False,)
print(scores_cont)


# %%
print(
    lxr.sentences_similarity(
        "India is going to be free soon", "soon we will be in india"
    )
)


# %%
