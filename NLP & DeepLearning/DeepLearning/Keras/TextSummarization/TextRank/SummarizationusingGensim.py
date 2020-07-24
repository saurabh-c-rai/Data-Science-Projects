#%%
from gensim.summarization.summarizer import summarize

# %%
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


# %%
print(summarize(text, ratio=0.3))

# %%
