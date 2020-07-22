#%%
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


#%%
system_summary = "Text summarization is an application of Natural Language Processing"

maunal_summary = (
    "Document summarization is one of the application under Natural Language Processing"
)
#%%
smoothing = SmoothingFunction()
#%%
reference = [system_summary.split()]
candidate = maunal_summary.split()
score_0 = sentence_bleu(reference, candidate, smoothing_function=smoothing.method0)
print(score_0)
score_1 = sentence_bleu(reference, candidate, smoothing_function=smoothing.method1)
print(score_1)
score_2 = sentence_bleu(reference, candidate, smoothing_function=smoothing.method2)
print(score_2)
score_3 = sentence_bleu(reference, candidate, smoothing_function=smoothing.method3)
print(score_3)
score_4 = sentence_bleu(reference, candidate, smoothing_function=smoothing.method4)
print(score_4)
score_5 = sentence_bleu(reference, candidate, smoothing_function=smoothing.method5)
print(score_5)
score_6 = sentence_bleu(reference, candidate, smoothing_function=smoothing.method6)
print(score_6)
score_7 = sentence_bleu(reference, candidate, smoothing_function=smoothing.method7)
print(score_7)

#%%
score = sentence_bleu(reference, candidate, weights=[0.3, 0.3, 0.3])
print(score)


# %%
