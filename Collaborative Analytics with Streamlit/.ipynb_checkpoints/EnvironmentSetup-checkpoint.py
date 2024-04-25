import time
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris



# -----------Basic Charts
# col_names = ['column 1','column 2','column 3',]

# data = pd.DataFrame(np.random.randint(30, size=(30, 3)), columns=col_names)

# 'line graph:'
# st.line_chart(data)

# 'bar graph:'
# st.bar_chart(data)


# -----------Line charts in Streamlit
# rows = np.random.randn(1,1)
# 'Growing Line Chart:'
# chart = st.line_chart(rows)
# for i in range(1,100):
#     new_rows = rows[0] + np.random.randn(1,1)
#     chart.add_rows(new_rows)
#     rows = new_rows
#     time.sleep(0.03)

# values = np.random.randn(10)
# 'matplotlibs Line Chart:'
# fig, ax = plt.subplots()
# ax.plot(values)
# st.pyplot(fig)

# -----------Line charts in Streamlit
# 'matpotlibs Bar Chart'
# animals = ['cat', 'cow', 'dog', 'goat']
# heights = [30,150,80,60]
# weights = [5,400,40,50]

# fig, ax = plt.subplots()

# x = np.arange(len(heights))
# width = 0.40

# ax.bar(x - 0.2, heights, width, color='#E5C4C1')
# ax.bar(x + 0.2, weights, width, color='#FFE5B4')

# ax.legend(['height', ['weight']])
# ax.set_xticks(x)
# ax.set_xticklabels(animals)

# st.pyplot(fig)

# 'matpotlibs Pie Chart'
# explode = [0.2,0.1,0.1,0.1]
# plot_pie, ax = plt.subplots()
# ax.pie(heights, explode = explode, labels = animals, autopct = '%1.1f%%', shadow=True)
# ax.axis('equal')

# st.pyplot(plot_pie)

# -----------Create statistical charts
iris_data = load_iris()
data = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)
data
'seaborn histplot'
fig = plt.figure()
sns.histplot(data=data, bins=(20))
st.pyplot(fig)

'seaborn boxplot'
fig = plt.figure()
sns.boxplot(data = data)
st.pyplot(fig)

'seaborn scatter'
fig = plt.figure()
sns.scatterplot(data = data)
st.pyplot(fig)


