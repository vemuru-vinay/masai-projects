import pandas as pd
import glob
#used glob for futire filename changing issues
files = glob.glob("data/trends*.json")
print(files[0])

df = pd.read_json(files[0])
#analyze the data
df.info()
df.describe()
df.shape
df.isna().sum()

#found out having duplicates and one null value
df.drop_duplicates(subset='post_id',inplace=True)

print(len(df))
#droped null values 
df.dropna(subset=['post_id','title','score','num_comments'], inplace=True)

print(len(df))

df.dtypes
#changed to integer
df['num_comments'] = df['num_comments'].astype(int)
df.dtypes

#removed cols having score > 5
df = df[df['score'] >= 5]


df['title'] = df['title'].str.strip()

#created a csv from df
df.to_csv("data/trends_clean.csv", index=False)

#no of stories in each category
df['category'].value_counts()





