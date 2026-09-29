
import pandas as pd
import numpy as np

#loaded data
df = pd.read_csv("data/trends_clean.csv")


df.shape


df.head()
#print avaeages of score and comments
print(f"Average of Score :{df['score'].mean():.2f}")
print(f"Average of num_comments :{df['num_comments'].mean():.2f}")
#printed a mean , median and standard devaition of score
print(f"Mean of score :{np.mean(df['score']):.2f}")
print(f"Median of score :{np.median(df['score']):.2f}")
print(f"std of score :{np.std(df['score']):.2f}")

#print max and min score 
print(f"max score: {np.max(df['score'])}")
print(f"min score: {np.min(df['score'])}")

#printed top category 
df['category'].value_counts().index[0]

df.head()

#printed a top commented story 
idx = df['num_comments'].idxmax()
df.loc[[idx], ['title','num_comments']]

#added two new cols
df['engagement'] = (df['num_comments']/( df['score']+1))

df['is_popular'] = np.where(df['score'] > np.mean(df['score']),True,False)

#saved new cleaned file
df.to_csv("data/trends_analyzed.csv")

print(f"Saved to data/trends_analysed.csv")
