# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
p_a = len(df[df['fico'] > 700])/len(df['fico'])
p_b = len(df[df['purpose'] == 'debt_consolidation'])/len(df['purpose'])
df1 = df[df['purpose'] == 'debt_consolation']
p_a_b = (p_a*p_b)/p_a
result = p_a_b ==  p_a
result

# code ends here


# --------------
# code starts here
#prob_lp = len(df[df['paid.back.loan']== 'Yes'])/len(df['paid.back.loan'])
#prob_cs = len(df[df['credit.policy']== 'Yes'])/len(df['credit.policy'])
#new_df = df[df['paid.back.loan']== 'Yes']
#prob_pd_cs = (prob_lp*prob_cs)/prob_lp
#bayes = (prob_pd_cs*prob_lp)/prob_cs
#print(bayes)

# code starts here
prob_lp=((df['paid.back.loan']=='Yes').sum())/len(df)
print(prob_lp)
prob_cs=((df['credit.policy']=='Yes').sum())/len(df)
print(prob_cs)
new_df=df[df['paid.back.loan']=='Yes']
prob_pd_cs=((new_df['credit.policy']=='Yes').sum())/len(new_df)
print(prob_pd_cs)
bayes=(prob_lp*prob_pd_cs)/(prob_cs)
print(bayes)

# code ends here


# --------------
# code starts here
df['purpose'].value_counts().plot.bar() 
df1 = df[df['paid.back.loan'] == 'No']
df1['purpose'].value_counts().plot.bar()
# code ends here


# --------------
# code starts here
inst_median = df['installment'].median()
inst_mean = df['installment'].mean()
df['installment'].plot.hist()
df['log.annual.inc'].plot.hist()

# code ends here


