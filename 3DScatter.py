from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

excel_data_df = pandas.read_csv('data\processedDataAll.csv', delimiter=';')

first_col = 'Page total likes'
second_col = 'Lifetime Post Total Impressions'
third_col = 'Lifetime Post reach by people who like your Page'

page_likes = excel_data_df[first_col].tolist()
total_impressions = excel_data_df[second_col].tolist()
post_share = excel_data_df[third_col].tolist()

post_like = excel_data_df['like'].tolist()

category_one_indexes = [index for index, x in enumerate(post_like) if x == 0]
category_two_indexes = [index for index, x in enumerate(post_like) if x == 1]
category_three_indexes = [index for index, x in enumerate(post_like) if x == 2]
category_four_indexes = [index for index, x in enumerate(post_like) if x == 3]
category_five_indexes = [index for index, x in enumerate(post_like) if x == 4]

categories_index = [category_one_indexes, category_two_indexes, category_three_indexes, category_four_indexes,
                    category_five_indexes]

# first_category
data_sets = []

for category in categories_index:
    page_like_category = [page_likes[index] for index in category]
    total_impressions_category = [total_impressions[index] for index in category]
    post_reach_category = [post_share[index] for index in category]
    category_list = [page_like_category, total_impressions_category, post_reach_category]
    data_sets.append(category_list)


# Creating figure
# fig = plt.figure(figsize=(16, 9))
ax = plt.axes(projection="3d")

# Add x, y gridlines
ax.grid(b=True, color='grey',
        linestyle='-.', linewidth=0.3,
        alpha=0.2)


colors = ['lime', 'purple', 'violet', 'darkorange', 'r']
markers = ['^', '<', '>', 'v', '+']

[data_sets[index].append(colors[index]) for index in range(5)]
[data_sets[index].append(markers[index]) for index in range(5)]

for category_data in data_sets:
    x = category_data[0]
    y = category_data[1]
    z = category_data[2]
    ax.scatter(x, y, z, marker=category_data[4], c=category_data[3])


plt.title("Post Likes distribution \n \n", fontweight='bold')
ax.tick_params(labelcolor='#3F3C3C', labelsize=8 )
ax.set_xlabel('\n \n  Page total likes')
ax.set_ylabel('\n \n    Lifetime Post \n Total Impressions')
ax.set_zlabel('\n \n   Lifetime Post reach \n by people who like your Page')

pop_a = mlines.Line2D([], [], color=colors[0], marker=markers[0], markersize=5, label='0-50 likes')
pop_b = mlines.Line2D([], [], color=colors[1], marker=markers[1], markersize=5, label='50-100 likes')
pop_c = mlines.Line2D([], [], color=colors[2], marker=markers[2], markersize=5, label='100-200 likes')
pop_d = mlines.Line2D([], [], color=colors[3], marker=markers[3], markersize=5, label='200-500 likes')
pop_e = mlines.Line2D([], [], color=colors[4], marker=markers[4], markersize=5, label='500+ likes')


plt.legend(handles=[pop_a,pop_b, pop_c, pop_d, pop_e], prop={'size': 6})


# show plot
plt.show()
