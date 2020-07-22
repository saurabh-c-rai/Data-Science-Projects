#%%
from rouge import Rouge

#%%
system_summary = "Text summarization is an application of Natural Language Processing"

maunal_summary = (
    "Document summarization is one of the application under Natural Language Processing"
)

#%%
rouge = Rouge()
scores = rouge.get_scores(system_summary, maunal_summary)
#%%
for key, value in scores[0].items():
    print(key, value)

