import pandas as pd
import matplotlib.pyplot as plt
import os


df = pd.read_csv('../data/trends_analyzed.csv')
df.head()

os.makedirs('../outputs')

top10 = df.sort_values('score', ascending=False).head(10)


top10


titles = top10['title'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)


titles


plt.barh(titles,top10['score'])
plt.xlabel('score')
plt.ylabel('titles')
plt.title("Top 10 Stories by Score")
plt.tight_layout()
plt.savefig('../outputs/chart1_top_stories.png')
plt.show()


category_counts = df['category'].value_counts()
category_counts


plt.figure()
plt.bar(category_counts.index, category_counts.values, color=['red', 'blue', 'green', 'orange', 'purple'])
plt.savefig('../outputs/chart2_top_categories.png')


popular = df[df['is_popular'] == True]
popular


not_popular = df[df['is_popular'] == False]
not_popular


plt.scatter(popular['score'], popular['num_comments'], color='green', label='Popular')
plt.xlabel('score')
plt.ylabel('num_comments')


plt.scatter(not_popular['score'], not_popular['num_comments'], color='red', label='Not Popular')


plt.scatter(popular['score'], popular['num_comments'], color='green', label='Popular')
plt.scatter(not_popular['score'], not_popular['num_comments'], color='red', label='Not Popular')
plt.savefig('../outputs/chart3_popularVSnot_popular.png')