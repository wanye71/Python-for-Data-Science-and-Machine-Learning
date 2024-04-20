# Python for Data Science and Machine Learning

## Table of Contents

1. [Matplotlib and Seaborn Libraries](#matplotlib-and-seaborn-libraries)
2. [Creating Standard Data Graphics](#creating-standard-data-graphics)
3. [Defining Elements of a Plot](#defining-elements-of-a-plot)
4. [Plot Formatting](#plot-formatting)
5. [Creating Labels and Annotations](#creating-labels-and-annotations)
6. [Visualizing Time Series](#visualizing-time-series)
7. [Creating Statistical Data Graphics in Seaborn](#creating-statistical-data-graphics-in-seaborn)

### Matplotlib and Seaborn Libraries
```python
# Sup, fam! We're about to spice up our cosmic data exploration with some visualizations:

# First up, we're importing the matplotlib library, our trusty tool for creating cosmic visualizations.
# It's like having a telescope to peer into the depths of our data universe and unveil its secrets.
import matplotlib.pyplot as plt

# Next, we're summoning the seaborn library, a powerful ally for creating stunning visualizations with ease.
# Seaborn adds an extra layer of style and flair to our cosmic charts, making them out of this world!
import seaborn as sns

# And last but not least, we're cherry-picking the DataFrame class from the pandas library.
# This class is like our essential tool for organizing and structuring our data for visualization.
from pandas import DataFrame

# Ayo, fam! We're about to create a cosmic DataFrame to explore the characteristics of cosmic beings:

# We're defining a dictionary 'data' containing information about cosmic beings:
# - 'names': Names of cosmic beings
# - 'age': Ages of cosmic beings
# - 'gender': Genders of cosmic beings
# - 'rank': Ranks of cosmic beings (maybe in a cosmic hierarchy)

data = {'names': ['steve', 'john', 'richard', 'sarah', 'randy', 'micheal', 'julie'],
        'age': [20, 22, 20, 21, 24, 23, 22],
        'gender': ['Male', 'Male', 'Male', 'Female', 'Male', 'Male', 'Female'],
        'rank': [2, 1, 4, 5, 3, 7, 6]}

# Next, we're using the DataFrame constructor to create a cosmic DataFrame 'df' from the 'data' dictionary.
# This DataFrame organizes the cosmic data into a structured format for exploration and analysis.
# It's like assembling a cosmic council of beings, each with their own characteristics and roles.
df = DataFrame(data)

# Get ready to explore the cosmic council and uncover the mysteries of cosmic beings!
df

### Matplotlib's Bar Chart
# Hey, fam! We're about to create a cosmic bar chart to compare the ages of cosmic beings:

# We're using the plt.bar() function to create a bar chart.
# This chart compares the ages of cosmic beings, with each bar representing a different being.
# It's like lining up the cosmic beings in a cosmic bar, with the height of each bar indicating their age.

plt.bar(df['names'], df['age'])

# Now, we're adding labels to the x-axis and y-axis to provide context to our cosmic bar chart.
# The x-axis label indicates the variable being compared (names of cosmic beings), while the y-axis label indicates the age.
plt.xlabel('Names')  # Label for the x-axis
plt.ylabel('Age')    # Label for the y-axis

# Next up, we're adding a title to our cosmic bar chart to give it a meaningful context.
# The title highlights the purpose of the chart, which is to compare the ages of cosmic beings.
plt.title('Compare Ages')  # Title of the chart

# Finally, we're using plt.show() to display our cosmic bar chart in all its glory!
plt.show()

### Seaborn's Bar Chart
# What up, fam! Get ready for some cosmic data visualization with seaborn:

# We're using the sns.barplot() function from the seaborn library to create a bar plot.
# This plot compares the ages of cosmic beings, with each bar representing a different being.
# It's like stacking up cosmic bars to showcase the ages of different beings.

plot = sns.barplot(data=df, x='names', y='age')

# Next, we're setting a title for our cosmic bar plot to give it a meaningful context.
# The title highlights the purpose of the plot, which is to compare the ages of cosmic beings.
plot.set_title("Comparing Ages")

# And there you have it! Our cosmic bar plot is ready to be unveiled and explored.
# Get ready to dive into the cosmic ages of beings!
plot;

### Line Plot Matplotlib
# Yo, fam! We're about to create a cosmic line plot to compare the ages of cosmic beings:

# We're using the plt.plot() function to create a line plot.
# This plot compares the ages of cosmic beings, with each point representing a different being.
# It's like connecting the cosmic dots to visualize the ages of different beings over a continuum.

plt.plot(df['names'], df['age'])

# Now, we're adding labels to the x-axis and y-axis to provide context to our cosmic line plot.
# The x-axis label indicates the variable being compared (names of cosmic beings), while the y-axis label indicates the age.
plt.xlabel('Names')  # Label for the x-axis
plt.ylabel('Age')    # Label for the y-axis

# Next up, we're adding a title to our cosmic line plot to give it a meaningful context.
# The title highlights the purpose of the plot, which is to compare the ages of cosmic beings.
plt.title('Compare Ages')  # Title of the plot

# Finally, we're using plt.show() to display our cosmic line plot in all its glory!
plt.show()

### Line Plot Seaborn
# Hey, fam! Get ready for some cosmic data visualization with seaborn:

# We're using the sns.lineplot() function from the seaborn library to create a line plot.
# This plot compares the ages of cosmic beings, with each point representing a different being.
# It's like connecting the cosmic dots to visualize the ages of different beings over a continuum.

plot = sns.lineplot(data=df, x='names', y='age')

# Next, we're setting a title for our cosmic line plot to give it a meaningful context.
# The title highlights the purpose of the plot, which is to compare the ages of cosmic beings.
plot.set_title('Compare Ages')

# And there you have it! Our cosmic line plot is ready to be unveiled and explored.
# Get ready to dive into the cosmic ages of beings!
plt.show()

### Pie Chart Matplotlib
# Brace yourselves, cosmic explorers! We're about to dive into the depths of data with a cosmic pie chart:

# Let's kick things off with the plt.pie() function, slicing and dicing the cosmic age pie into segments.
# Each slice represents a cosmic being, giving us a visual feast of age proportions.

plt.pie(df['age'], labels=df['names'])

# Now, let's add a touch of cosmic flair with a title that sets the stage for our cosmic journey.
# "Age Comparison" serves as our guiding star, illuminating the purpose of our cosmic chart.

plt.title('Age Comparison')  # Title of the chart

# And voilà! Our cosmic pie chart is ready to take center stage, captivating the minds of cosmic wanderers.
# Prepare to embark on a visual odyssey through the age proportions of cosmic beings!
plt.show()

### Pie Chart Seaborn
# Hold onto your space helmets, fam! We're about to add a splash of color to our cosmic pie chart:

# First, we're grabbing a palette of pastel colors using the sns.color_palette() function.
# These colors will breathe life into our cosmic pie chart, making each slice pop with vibrancy.

colors = sns.color_palette("pastel")[0:50]

# Now, armed with our cosmic color palette, we're ready to create our pie chart using plt.pie().
# Each slice will be adorned with a unique pastel hue, adding a visual treat to our cosmic exploration.

plt.pie(df['age'], labels=df['names'], colors=colors)

# And there you have it! Our cosmic pie chart is now a kaleidoscope of colors, ready to mesmerize and delight.
# Get ready to feast your eyes on the cosmic spectrum of age proportions among beings!
plt.show()
```

### Creating Standard Data Graphics
```python
# Rev up those engines, gearheads! We're importing the tools we need for a high-octane data exploration:

# First up, we're bringing in numpy to supercharge our numerical computations.
# Think of numpy as the engine under the hood, providing raw power for our data manipulations and calculations.
import numpy as np

# Next, we're calling in pandas to steer us through the winding roads of data management.
# With pandas, we'll cruise through datasets like seasoned drivers, effortlessly handling rows, columns, and indices.
import pandas as pd

# Now, let's invite matplotlib.pyplot to the party and light up the dashboard with visualizations.
# This versatile toolkit will transform our data into stunning charts and graphs, painting a vivid picture of our insights.
import matplotlib.pyplot as plt

# Last but not least, we're gearing up with some specific imports to fine-tune our ride.
# We'll grab randn from numpy.random for generating random numbers,
# Series and DataFrame from pandas to structure our data like pro racers,
# and rcParams from matplotlib to customize the look and feel of our visualizations.

# Importing randn function for generating random numbers
from numpy.random import randn  

# Importing Series and DataFrame for data manipulation
from pandas import Series, DataFrame  

 # Importing rcParams for customizing plot parameters
from matplotlib import rcParams

# Alright, gearheads! With our toolkit fully loaded, it's time to hit the data highways and explore the fast lane of analytics.

### Creating a line chart from a list object
x = range(1, 10)
y = [1,2,3,4,0,4,3,2,1]
plt.plot(x, y)
# Alright, gearheads! Let's dive into our data exploration journey with some hot wheels:

# First up, we're setting the path to our dataset. It's like plotting the coordinates on our map to the treasure chest of data.
address = '/workspaces/Python-for-Data-Science-and-Machine-Learning/mtcars.csv'

# Next, we're loading our dataset into the garage of our analysis using pandas.read_csv().
# Think of this step as driving our data into the workshop, ready to be inspected and fine-tuned.
cars = pd.read_csv(address)

# Now that our dataset is parked in the garage, it's time to give the columns some slick new names.
# We're renaming the columns to make them more intuitive and easier to navigate on our data highway.
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']

# With our dataset prepped and ready, let's rev up the engine and extract the horsepower data.
# We're selecting the 'hp' column from our dataset and assigning it to the variable 'mpg'.
mpg = cars['hp']
# Alright, gearheads! Let's buckle up and hit the road with some data visualization:

# We're revving up the engine of data exploration by plotting the 'mpg' (miles per gallon) data.
# This line of code creates a line plot of the 'mpg' column, showcasing the trend of fuel efficiency across different cars.

mpg.plot()

# With the plot fired up, we're ready to cruise through the data highway and uncover insights about fuel efficiency.
# Get ready to ride the waves of data visualization, gearheads!
# Alright, gearheads! Let's gear up for a multi-dimensional data exploration:

# We're creating a DataFrame 'df' to focus on specific features of interest: 'cyl' (cylinders), 'wt' (weight), and 'mpg' (miles per gallon).
# Think of this DataFrame as our custom-built ride, equipped with just the right features for our data exploration journey.
df = cars[['cyl', 'wt', 'mpg']]

# Now that our custom ride is ready, let's hit the road and visualize the data.
# This line of code creates a line plot for each column in the DataFrame 'df', showcasing the trends and relationships between features.

df.plot();
### Creating bar charts
#### Creating a bar chart from a list
# Alright, gearheads! Let's fire up the engines and hit the data highway with some bar plots:

# We're using matplotlib.pyplot's bar function to create a bar plot.
# This function requires two arguments: 'x' and 'y', representing the x-axis and y-axis data respectively.
# Think of this as fueling up our chart with the data needed to construct the bars.

plt.bar(x, y)

# With the bars fueled up and ready to go, it's time to hit the road and visualize the data.
# Get ready to cruise through the twists and turns of data exploration with this bar plot, gearheads!

# Note: Don't forget to customize the labels, titles, and other plot parameters to make your visualization shine.

#### Creating a bar chart from Pandas objects
# Alright, gearheads! Let's fire up the engines and hit the data highway with some bar plots:

# We're using the plot function on the 'mpg' Series with the kind parameter set to 'bar'.
# This tells pandas to create a bar plot of the 'mpg' data.
# Think of this as fueling up our chart with the 'mpg' data and choosing the 'bar' style for visualization.

mpg.plot(kind='bar')

# With the bars fueled up and ready to go, it's time to hit the road and visualize the data.
# Get ready to cruise through the twists and turns of data exploration with this bar plot, gearheads!

# Note: Don't forget to customize the labels, titles, and other plot parameters to make your visualization shine.

# Alright, gearheads! Let's fire up the engines and hit the data highway with some horizontal bar plots:

# We're using the plot function on the 'mpg' Series with the kind parameter set to 'barh'.
# This tells pandas to create a horizontal bar plot of the 'mpg' data.
# Think of this as fueling up our chart with the 'mpg' data and choosing the 'barh' style for visualization.

mpg.plot(kind='barh')

# With the bars fueled up and ready to go, it's time to hit the road and visualize the data.
# Get ready to cruise through the twists and turns of data exploration with this horizontal bar plot, gearheads!

# Note: Don't forget to customize the labels, titles, and other plot parameters to make your visualization shine.

### Creating a pie chart
# Alright, gearheads! Let's buckle up and hit the data highway with a tasty pie chart:

# We're using matplotlib.pyplot's pie function to create a pie chart.
# This function requires one argument, 'x', representing the data to be visualized.
# Think of this as fueling up our chart with the data needed to construct the slices of the pie.

plt.pie(x)

# With the slices of the pie fueled up and ready to go, it's time to hit the road and visualize the data.
# Get ready to savor the flavors of data exploration with this delicious pie chart, gearheads!

# Note: Don't forget to customize the labels, colors, and other plot parameters to make your visualization shine.

### Saving a plot
# Alright, gearheads! Let's buckle up and hit the data highway with a tasty pie chart:

# We're using matplotlib.pyplot's pie function to create a pie chart.
# This function requires one argument, 'x', representing the data to be visualized.
# Think of this as fueling up our chart with the data needed to construct the slices of the pie.

plt.pie(x)

# With the slices of the pie fueled up and ready to go, it's time to hit the road and visualize the data.
# Get ready to savor the flavors of data exploration with this delicious pie chart, gearheads!

# Now, let's save the pie chart as an image file.
# We're using the savefig function to save the current figure (pie chart) as a JPEG file named 'pie_chart.jpg'.
# Think of this as taking a snapshot of our chart and storing it for later use.

plt.savefig('pie_chart.jpg')

# Finally, let's display the pie chart on the screen.
# We use the show function to render the pie chart in our current environment.

plt.show()
```

### Defining Elements of a Plot
```python
# Alright, gearheads! Let's kick off our data exploration journey by importing the necessary tools:

# First up, we're bringing in numpy to supercharge our numerical computations.
# Think of numpy as the engine under the hood, providing raw power for our data manipulations and calculations.
import numpy as np

# Next, we're importing the randn function from numpy.random.
# This function generates random numbers from a standard normal distribution.
# Think of this as fueling up our dataset with some randomness for added excitement.

from numpy.random import randn

# Now, let's invite pandas to the party and light up the dashboard with data management capabilities.
# With pandas, we'll cruise through datasets like seasoned drivers, effortlessly handling rows, columns, and indices.
import pandas as pd

# Next, we're importing Series and DataFrame from pandas.
# These data structures will be our trusty companions on the data exploration journey, helping us organize and analyze our data.
from pandas import Series, DataFrame

# Now, it's time to bring in matplotlib.pyplot to visualize our insights.
# This versatile toolkit will transform our data into stunning charts and graphs, painting a vivid picture of our analyses.
import matplotlib.pyplot as plt

# Last but not least, we're gearing up with rcParams from matplotlib to customize the look and feel of our visualizations.
# With rcParams, we'll fine-tune our plots to perfection, ensuring they match our style and preferences.
from matplotlib import rcParams

# Alright, gearheads! With our toolkit fully loaded, it's time to hit the data highways and explore the fast lane of analytics.

%matplotlib inline

# Alright, gearheads! Let's fine-tune the dimensions of our plots to ensure they pack a punch:

# We're customizing the figure size using rcParams.
# This line of code sets the width and height of our plots to 5 inches and 4 inches respectively.
# Think of this as adjusting the canvas size before we start painting our data masterpiece.

rcParams['figure.figsize'] = 5, 4

# With the figure dimensions dialed in, our plots are ready to shine on the data highway.
# Get ready to showcase your insights with style and precision, gearheads!

# Note: You can tweak the width and height values to suit your preferences and display requirements.

### Defining axes, ticks, and grids
# Alright, gearheads! Let's take control of our plot axes and steer our visualization in the right direction:

# We're setting up the data for our plot.
# 'x' represents the x-axis values, while 'y' represents the corresponding y-axis values.
# Think of these as the coordinates that define the path of our data journey.

x = range(1, 10)
y = [1, 2, 3, 4, 0, 4, 3, 2, 1]

# Now, it's time to create a figure to hold our plot.
# We call plt.figure() to initialize a new figure object.
# This figure will serve as the canvas where we'll visualize our data insights.
fig = plt.figure()

# Next, we're adding axes to our figure using fig.add_axes().
# The [left, bottom, width, height] parameters define the position and size of the axes within the figure.
# Think of this as defining the space where our plot will be displayed.
# Here, we're placing the axes at (0.1, 0.1) coordinates with a width and height of 1 unit each.

ax = fig.add_axes([0.1, 0.1, 1, 1])

# With the axes set up, it's time to plot our data.
# We use ax.plot() to create a line plot of 'x' against 'y' within the specified axes.
# Think of this as laying out the path of our data journey on the canvas.

ax.plot(x, y)

# With our plot ready to roll, it's time to hit the road and explore the twists and turns of data visualization.
# Get ready to navigate through the data highway with precision and style, gearheads!

# Note: You can adjust the position and size of the axes to customize the layout of your plot.

# Alright, gearheads! Let's fine-tune the axes of our plot for precision navigation through the data highway:

# We're setting up the data for our plot.
# 'x' represents the x-axis values, while 'y' represents the corresponding y-axis values.
# Think of these as the coordinates that define the path of our data journey.

x = range(1, 10)
y = [1, 2, 3, 4, 0, 4, 3, 2, 1]

# Now, it's time to create a figure to hold our plot.
# We call plt.figure() to initialize a new figure object.
# This figure will serve as the canvas where we'll visualize our data insights.

fig = plt.figure()

# Next, we're adding axes to our figure using fig.add_axes().
# The [left, bottom, width, height] parameters define the position and size of the axes within the figure.
# Think of this as defining the space where our plot will be displayed.

ax = fig.add_axes([0.1, 0.1, 1, 1])

# With the axes set up, let's customize the limits of the x and y axes for a focused view of our data.
# We use ax.set_xlim() and ax.set_ylim() to set the lower and upper bounds for the x and y axes respectively.

ax.set_xlim([1, 9])  # Setting the x-axis limits from 1 to 9
ax.set_ylim([0, 5])  # Setting the y-axis limits from 0 to 5

# Now, let's customize the tick marks on the x and y axes to ensure clear navigation through the plot.
# We use ax.set_xticks() and ax.set_yticks() to specify the locations of the tick marks.

ax.set_xticks([0, 1, 2, 3, 4, 5, 8, 9, 10])  # Setting tick marks on the x-axis
ax.set_yticks([0, 1, 2, 3, 4, 5])           # Setting tick marks on the y-axis

# With the axes limits and tick marks dialed in, it's time to plot our data and hit the road!
# Get ready to navigate through the twists and turns of data exploration with precision and style, gearheads!

# Note: You can adjust the axes limits and tick marks to suit your preferences and display requirements.
ax.plot(x,y)

# Alright, gearheads! Let's add some gridlines to our plot to stay on track and navigate the data highway with precision:

# We're setting up the data for our plot.
# 'x' represents the x-axis values, while 'y' represents the corresponding y-axis values.
# Think of these as the coordinates that define the path of our data journey.

x = range(1, 10)
y = [1, 2, 3, 4, 0, 4, 3, 2, 1]

# Now, it's time to create a figure to hold our plot.
# We call plt.figure() to initialize a new figure object.
# This figure will serve as the canvas where we'll visualize our data insights.

fig = plt.figure()

# Next, we're adding axes to our figure using fig.add_axes().
# The [left, bottom, width, height] parameters define the position and size of the axes within the figure.
# Think of this as defining the space where our plot will be displayed.

ax = fig.add_axes([0.1, 0.1, 1, 1])

# With the axes set up, let's customize the limits of the x and y axes for a focused view of our data.
# We use ax.set_xlim() and ax.set_ylim() to set the lower and upper bounds for the x and y axes respectively.

ax.set_xlim([1, 9])  # Setting the x-axis limits from 1 to 9
ax.set_ylim([0, 5])  # Setting the y-axis limits from 0 to 5

# Now, it's time to add gridlines to our plot for better visualization.
# We use ax.grid() to activate the gridlines on the plot.

ax.grid()

# Finally, let's plot our data and hit the road!
# We use ax.plot() to create a line plot of 'x' against 'y' within the specified axes.
# This command lays out the path of our data journey on the canvas.

ax.plot(x, y)

# With the gridlines in place and our data plotted, it's time to rev up the engine and embark on our data exploration adventure!

### Generating multiple plots in one figure with subplots
fig = plt.figure()
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(x)
ax2.plot(x,y)

# Note: You can customize the appearance of the gridlines using additional parameters in ax.grid() if needed.
# Example:
        # plt.xlabel('X Axis Label')          # Customize X axis label
        # plt.ylabel('Y Axis Label')          # Customize Y axis label
        # plt.title('Customized Plot Title')  # Customize plot title
        # plt.xticks(rotation=45)             # Rotate X axis tick labels for better readability
        # plt.yticks(color='blue')            # Change color of Y axis tick labels

        # plt.show()
```

### Plot Formatting
```python

```

### Creating Labels and Annotations
```python

```

### Visualizing Time Series
```python

```

### Creating Statistical Data Graphics in Seaborn
```python

```






