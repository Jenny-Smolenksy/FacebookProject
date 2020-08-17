from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

excel_data_df = pandas.read_csv('dataset_Facebook.csv').to_numpy()

first_col = 'Page total likes'
second_col = 'Lifetime Post Total Impressions'
third_col = 'Lifetime Post reach by people who like your Page'

page_likes = excel_data_df[first_col].tolist()
total_impressions = excel_data_df[second_col].tolist()
post_reach = excel_data_df[third_col].tolist()



post_type = excel_data_df['Type'].tolist()
type_to_number_dictionary = dict(Link=1, Photo=2, Status=3, Video=4)
post_type = np.array([type_to_number_dictionary.get(key) for key in post_type])



post_like = excel_data_df['like'].tolist()

category_one_indexes = [post_like.index(sample) for sample in post_like if 0 <= sample < 50]
category_two_indexes = [post_like.index(sample) for sample in post_like if 50 <= sample < 100]
category_three_indexes = [post_like.index(sample) for sample in post_like if 100 <= sample < 200]
category_four_indexes = [post_like.index(sample) for sample in post_like if 200 <= sample < 500]
category_five_indexes = [post_like.index(sample) for sample in post_like if sample >= 500]

categories_index = [category_one_indexes, category_two_indexes, category_three_indexes, category_four_indexes,
                    category_five_indexes]

# first_category
data_sets = []

for category in categories_index:
    page_like_category = [page_likes[index] for index in category]
    total_impressions_category = [total_impressions[index] for index in category]
    post_type_category = [post_type[index] for index in category]
    post_reach_category = [post_reach[index] for index in category]
    category_list = [page_like_category, total_impressions_category, post_reach_category]
    data_sets.append(category_list)

# Creating dataset
z = [10, 100, 20, 300, 30, 40]
x = [10, 100, 20, 300, 30, 40]
y = [10, 100, 20, 300, 30, 40]

# Creating figure
# fig = plt.figure(figsize=(16, 9))
ax = plt.axes(projection="3d")

# Add x, y gridlines
ax.grid(b=True, color='grey',
        linestyle='-.', linewidth=0.3,
        alpha=0.2)

# Creating color map
# my_cmap = plt.get_cmap('hsv')

xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ys = [5, 6, 2, 3, 13, 4, 1, 2, 4, 8]
zs = [2, 3, 3, 3, 5, 7, 9, 11, 9, 10]

xt = [-1, -2, -3, -4, -5, -6, -7, 8, -9, -10]
yt = [-5, -6, -2, -3, -13, -4, -1, 2, -4, -8]
zt = [-2, -3, -3, -3, -5, -7, 9, -11, -9, -10]

# ax.scatter(xs, ys, zs, c='r', marker='o')
# ax.scatter(xt, yt, zt, c='b', marker='^')

colors = ['lime', 'purple', 'violet', 'darkorange', 'r']
markers = ['^', '<', '>', 'v', '+']

[data_sets[index].append(colors[index]) for index in range(5)]
[data_sets[index].append(markers[index]) for index in range(5)]

for category_data in data_sets:
    x = category_data[0]
    y = category_data[1]
    z = category_data[2]
    ax.scatter(x, y, z, marker=category_data[4], c=category_data[3])

# Creating plot
# sctt = ax.scatter3D(x, y, z,
#                    c='r',
#                    marker='^')

# plt.title("simple 3D scatter plot")
ax.set_xlabel(first_col, fontweight='bold')
ax.set_ylabel(second_col, fontweight='bold')
ax.set_zlabel(third_col, fontweight='bold')
# fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)




pop_a = mlines.Line2D([], [], color=colors[0], marker=markers[0], markersize=5, label='0-50 likes')
pop_b = mlines.Line2D([], [], color=colors[1], marker=markers[1], markersize=5, label='50-100 likes')
pop_c = mlines.Line2D([], [], color=colors[2], marker=markers[2], markersize=5, label='100-200 likes')
pop_d = mlines.Line2D([], [], color=colors[3], marker=markers[3], markersize=5, label='200-500 likes')
pop_e = mlines.Line2D([], [], color=colors[4], marker=markers[4], markersize=5, label='500+ likes')


plt.legend(handles=[pop_a,pop_b, pop_c, pop_d, pop_e], prop={'size': 6})


# show plot
plt.show()
