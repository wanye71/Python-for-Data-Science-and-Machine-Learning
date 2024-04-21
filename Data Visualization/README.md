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
# Ayo, we're about to kick it off with some data exploration using seaborn. Think of seaborn as the slick, electric ride that's gonna take us on a data adventure like no other.
import seaborn as sb
# Next up, we're bringing in numpy, the powerhouse under the hood of our data vehicle. It's like the electric motor that gives us the horsepower to crunch numbers lightning fast.
import numpy as np
# Pandas is joining the party, and it's our go-to for handling tabular data. Picture it as the supercharged battery pack that keeps our data vehicle juiced up and ready to roll.
import pandas as pd
# Time to bring in matplotlib.pyplot, the graphic interface of our data vehicle. It's like the flashy dashboard that displays our data insights with style and flair.
import matplotlib.pyplot as plt

# We're also importing Series and DataFrame from pandas, essential tools in our data exploration toolkit. Series is like the sleek headlights that illuminate individual data points, while DataFrame is the spacious cabin where all our data can chill together.
from pandas import Series, DataFrame
# Lastly, rcParams from pylab is coming in to customize the look and feel of our data visualizations. It's like adding custom rims and paint jobs to our data vehicle, making sure it stands out on the data highway.
from pylab import rcParams

# First up, we're enabling inline plotting with `%matplotlib inline`. This 
# magic command ensures that any plots we create will be displayed directly 
# within our Jupyter Notebook or other IPython environment.
%matplotlib inline

# Next, we're setting the default figure size for our plots using 
# `rcParams['figure.figsize'] = 5, 4`. This line adjusts the width and height 
# of our plots to be 5 inches by 4 inches, ensuring they're the perfect size 
# for viewing and sharing.
rcParams['figure.figsize'] = 5, 4

# Now, we're setting the style of our plots to 'whitegrid' with 
# `sb.set_style('whitegrid')`. This style creates a clean and minimalist 
# background with horizontal gridlines, making our plots easy to read and 
# understand.
sb.set_style('whitegrid')

### Defining plot color
# We're defining the x values using the range function, starting from 1 and 
# ending at 9 (exclusive). This creates a sequence of integers from 1 to 9.
x = range(1, 10)

# Next, we're defining the y values as a list containing numeric values. This 
# list represents the heights of the bars in our bar plot.
y = [1, 2, 3, 4, 0.5, 4, 3, 2, 1]

# Now, we're creating a bar plot using the `plt.bar()` function from 
# matplotlib.pyplot. This function takes the x and y values as input and 
# generates a bar plot with x on the horizontal axis and y on the vertical 
# axis.
plt.bar(x, y)

# We're defining a list called 'wide' containing the widths of each bar in the 
# bar plot. These widths will determine the size of each bar on the x-axis.
wide = [0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5]

# We're also defining a list called 'color' containing a single color string 
# 'salmon'. This will set the color of all bars in the bar plot to salmon.
color = ['salmon']

# Now, we're creating a bar plot using the `plt.bar()` function from 
# matplotlib.pyplot. In addition to the x and y values, we're specifying 
# 'width=wide' to set the widths of the bars, 'color=color' to set the color 
# of the bars to salmon, and 'align='center'' to align the bars with the 
# center of each x-coordinate.
plt.bar(x, y, width=wide, color=color, align='center')

# We're defining a variable called 'address' and assigning it the file path of 
# the CSV file containing the data. This path points to the location of the 
# file on the file system.
address = '/workspaces/Python-for-Data-Science-and-Machine-Learning/mtcars.csv'

# We're using the `pd.read_csv()` function from the pandas library to read the 
# data from the CSV file located at the specified address. The data is stored 
# in a DataFrame called 'cars'.
cars = pd.read_csv(address)

# We're renaming the columns of the 'cars' DataFrame using the `columns` 
# attribute. The new column names are specified as a list containing strings.
# These names are assigned to the columns in the order they appear in the list.
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 
                'vs', 'am', 'gear', 'carb']

# We're selecting the 'mpg' column from the 'cars' DataFrame using bracket 
# indexing. This creates a Series object containing the miles per gallon (mpg) 
# values for each car in the dataset.
mpg = cars['mpg']

# We're using the `plot()` method on the 'mpg' Series to create a line plot of 
# the mpg values. The x-axis will automatically use the index of the Series, 
# while the y-axis will represent the mpg values.
mpg.plot()

# We're selecting a subset of columns from the 'cars' DataFrame using double 
# square brackets '[[]]'. This syntax allows us to select multiple columns 
# simultaneously. The selected columns are 'cyl', 'mpg', and 'wt'.
df = cars[['cyl', 'mpg', 'wt']]

# We're using the `plot()` method on the DataFrame 'df' to create a line plot. 
# By default, the index of the DataFrame will be used for the x-axis, and each 
# column will be plotted as a separate line on the y-axis.
df.plot()

# We're defining a list called 'color_theme' containing three color strings: 
# 'teal', 'peachpuff', and 'powderblue'. These colors will be used to customize 
# the line colors in the plot.
color_theme = ['teal', 'peachpuff', 'powderblue']

# We're using the `plot()` method on the DataFrame 'df' to create a line plot. 
# Additionally, we're specifying the 'color' parameter and passing the 
# 'color_theme' list. This parameter assigns the specified colors to the lines 
# in the plot, in the order they appear in the list.
df.plot(color=color_theme)
# We're defining a list called 'z' containing numeric values. This list 
# represents the data that will be visualized in the pie chart.
z = [1, 1, 0.1, 2, 0.5]

# We're using the `plt.pie()` function from matplotlib.pyplot to create a pie 
# chart. The 'z' list is passed as the first argument, providing the data to be 
# visualized. Additionally, we're specifying the 'explode' parameter with a list 
# [0, 0.1, 0, 0.1, 0]. This parameter controls the "explode" effect, where 
# slices are separated from the rest of the pie. A value of 0 means no 
# separation, while a non-zero value creates a separation.
plt.pie(z, explode=[0, 0.1, 0, 0.1, 0])

# We're displaying the pie chart using the `plt.show()` function.
plt.show()

# We're defining a list called 'color_theme' containing hexadecimal color 
# values. These colors will be used to customize the colors of the slices in 
# the pie chart.
color_theme = ["#E0BBE4", "#957DAD", "#D291BC", "#FEC8D8", "#FFDFD3"]

# We're using the `plt.pie()` function from matplotlib.pyplot to create a pie 
# chart. The 'z' list is passed as the first argument, providing the data to be 
# visualized. Additionally, we're specifying the 'colors' parameter and passing 
# the 'color_theme' list. This parameter assigns the specified colors to the 
# slices in the pie chart, in the order they appear in the list.
plt.pie(z, colors=color_theme)

# We're displaying the pie chart using the `plt.show()` function.
plt.show()

### Customize Line Styles
# We're defining a range of values from 1 to 10 (excluding 10) and assigning it 
# to the variable 'x1'. This range will be used as the x-values for the second 
# line plot.
x1 = range(1, 10)

# We're defining a list 'y1' containing numeric values representing the 
# y-coordinates for the second line plot.
y1 = [9, 8, 7, 6, 5, 4, 3, 2, 1]

# We're using the `plt.plot()` function from matplotlib.pyplot to create the 
# first line plot. The 'x' and 'y' lists are passed as arguments, providing 
# the x and y coordinates of the points to be connected by lines.
plt.plot(x, y)

# We're using the `plt.plot()` function again to create the second line plot. 
# This time, the 'x1' and 'y1' lists are passed as arguments to specify the 
# coordinates of the points for the second line plot.
plt.plot(x1, y1)

# We're using the `plt.plot()` function from matplotlib.pyplot to create the 
# first line plot. The 'x' and 'y' lists are passed as arguments, providing 
# the x and y coordinates of the points to be connected by lines. Additionally, 
# we're specifying the 'ls' parameter with the value 'solid' to set the line 
# style to solid and the 'lw' parameter with the value 5 to set the line width 
# to 5.
plt.plot(x, y, ls='solid', lw=5)

# We're using the `plt.plot()` function again to create the second line plot. 
# This time, the 'x1' and 'y1' lists are passed as arguments to specify the 
# coordinates of the points for the second line plot. Additionally, we're 
# specifying the 'ls' parameter with the value '--' to set the line style to 
# dashed and the 'lw' parameter with the value 10 to set the line width to 10.
plt.plot(x1, y1, ls='--', lw=10)

plt.plot(x,y, marker='1', mew=20)
plt.plot(x1,y1, marker='+', mew=15)

```

### Creating Labels and Annotations
```python

# We're importing the NumPy library and aliasing it as 'np'. NumPy is like 
# the reliable friend who always has your back, providing support for arrays, 
# matrices, and mathematical functions.
import numpy as np

# We're importing the Pandas library and aliasing it as 'pd'. Pandas is like 
# your loyal companion, helping you manage and analyze your data with ease.
import pandas as pd

# We're importing the Series and DataFrame classes from the Pandas library. 
# These classes are like the dynamic duo, representing one-dimensional 
# labeled arrays (Series) and two-dimensional labeled data structures 
# (DataFrame), respectively.
from pandas import Series, DataFrame

# We're importing the pyplot module from the Matplotlib library and aliasing 
# it as 'plt'. Matplotlib is like the artist, allowing us to create beautiful 
# visualizations to showcase our data.
import matplotlib.pyplot as plt

# We're importing the rcParams module from the pylab library. rcParams is like 
# the matchmaker, helping us customize the default settings for plotting to 
# create stunning visual displays.
from pylab import rcParams

# We're importing the seaborn library and aliasing it as 'sns'. Seaborn is like 
# the cupid of data visualization, adding a touch of charm and elegance to our 
# plots with its high-level interface for drawing attractive statistical graphics.
import seaborn as sns

# Ah, preparing the stage for our romantic rendezvous with data visualization!

# We're using `%matplotlib inline` to ensure that Matplotlib plots are displayed 
# directly within the Jupyter Notebook, creating a seamless and immersive 
# experience for our audience.
%matplotlib inline

# We're setting the figure size using `rcParams['figure.figsize']` to create a 
# spacious and inviting atmosphere for our visualizations. Just like preparing 
# a cozy setting for a romantic dinner, this ensures our plots have enough room 
# to shine.
rcParams['figure.figsize'] = 15, 4



# We're setting the style of the plots using `sns.set_style()`. By choosing the 
# 'darkgrid' style, we're adding a touch of mystery and allure to our visual 
# storytelling. It's like setting the mood with dim lighting and a hint of 
# suspense, keeping our audience captivated throughout the journey.
sns.set_style('darkgrid')

### Labeling plot features
#### The functional method
# Ah, the anticipation of revealing our data's story through 
# the artistry of bar plots!

# We're defining our x-values using the `range()` function, 
# creating a sequence of numbers from 1 to 9. It's like 
# selecting the perfect background music to set the mood for 
# our romantic evening.
x = range(1, 10)

# We're defining our y-values, representing the heights of 
# the bars in our bar plot. Each value corresponds to a 
# romantic gesture, ranging from heartfelt declarations to 
# subtle whispers of affection.
y = [1, 2, 3, 4, 0.5, 4, 3, 2, 1]

# We're creating a bar plot using `plt.bar()`, where each 
# bar represents a unique aspect of our data. It's like 
# arranging a bouquet of flowers, with each bloom 
# symbolizing a different emotion in our romantic journey.
plt.bar(x, y)

# We're adding a label to the x-axis using `plt.xlabel()`, 
# providing context for the values displayed along the 
# horizontal axis. It's like adding a caption to a cherished 
# photograph, helping us remember the special moments 
# captured in our data.
plt.xlabel('The x-axis label')

# We're adding a label to the y-axis using `plt.ylabel()`, 
# conveying the meaning behind the values represented along 
# the vertical axis. It's like expressing the depth of our 
# feelings, ensuring our audience understands the significance 
# of each data point in our romantic narrative.
plt.ylabel('The y-axis label')


z = [1,2,3,4,0.5]
veh_type = ['bicycle', 'motorbike', 'car', 'van', 'stroller']

# We're creating a pie chart using `plt.pie()`, where each slice 
# represents a different vehicle type. It's like exploring a 
# diverse range of transportation options, from bicycles to strollers.
plt.pie(z, labels=veh_type)
plt.show()

### The object-oriented method

# We're reading data from a CSV file containing information about cars.
address = '/workspaces/Python-for-Data-Science-and-Machine-Learning/mtcars.csv'
cars = pd.read_csv(address)

# We're renaming the columns of the DataFrame to enhance readability 
# and comprehension. It's like giving each car in our dataset a 
# distinctive name for easier identification.
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'dratr', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']

# We're selecting the 'mpg' column from the DataFrame, representing 
# the miles per gallon for each car. It's like focusing on a key 
# aspect of each romantic partner's personality.
mpg = cars.mpg

# We're creating a new figure object for our plot.
fig = plt.figure()

# We're adding an axes object to the figure, specifying its position 
# and dimensions. It's like setting up a canvas for our artistic 
# expression.
ax = fig.add_axes([.1,.1,1,1])

# We're plotting the 'mpg' values on the axes. It's like sketching 
# the outline of our romantic journey, highlighting the fuel efficiency 
# of each car.
mpg.plot()

# We're setting the positions of the x-axis ticks and customizing their 
# labels. It's like marking significant milestones in our journey and 
# attaching meaningful memories to each one.
ax.set_xticks(range(32))
ax.set_xticklabels(cars['car_names'], rotation=60, fontsize='medium')

# We're setting the title of the plot to describe its content. It's like 
# giving our artwork a title that captures its essence and emotional impact.
ax.set_title('Miles Per Gallon of Cars in mtcars')

# We're adding labels to the x-axis and y-axis to provide context for the 
# data being displayed. It's like adding captions to our artwork to guide 
# the viewer's interpretation.
ax.set_xlabel('car names')
ax.set_ylabel('miles/gal')

## Adding a legend to your plot
### The functional method
# We're creating another pie chart using `plt.pie()` without specifying 
# data labels. It's like exploring different aspects of our romantic journey 
# without attaching specific meanings to each slice.
plt.pie(z)
plt.legend(veh_type, loc='best')
plt.show()

### The object-oriented method

# We're creating another figure object for our plot.
fig = plt.figure()
ax = fig.add_axes([.1,.1,1,1])

# We're plotting the 'mpg' values on the axes. It's like expressing 
# the fuel efficiency of each car through visual representation.
mpg.plot()

# We're setting the positions of the x-axis ticks and customizing their 
# labels. It's like marking significant milestones in our journey and 
# attaching meaningful memories to each one.
ax.set_xticks(range(32))
ax.set_xticklabels(cars.car_names, rotation=60, fontsize='medium')

# We're setting the title of the plot to describe its content. It's like 
# giving our artwork a title that captures its essence and emotional impact.
ax.set_title('Miles Per Gallon of Cars in mtcars')

# We're adding labels to the x-axis and y-axis to provide context for the 
# data being displayed. It's like adding captions to our artwork to guide 
# the viewer's interpretation.
ax.set_xlabel('car names')
ax.set_ylabel('miles/gal')

# We're adding a legend to the plot to provide additional information about 
# the displayed data. It's like providing a key to help the viewer interpret 
# the visual elements in our artwork.
ax.legend(loc='best')

### Annotating your plot

# We're finding the maximum 'mpg' value in the dataset. It's like identifying 
# the peak of our romantic journey, the moment of maximum fuel efficiency.
mpg.max()

# We're creating another figure object for our plot.
fig = plt.figure()
ax = fig.add_axes([.1,.1,1,1])

# We're plotting the 'mpg' values on the axes. It's like painting the canvas 
# of our romantic journey with strokes of data points, visualizing the 
# variability in fuel efficiency.
mpg.plot()

# We're setting the positions of the x-axis ticks and customizing their 
# labels. It's like marking significant milestones in our journey and 
# attaching meaningful memories to each one.
ax.set_xticks(range(32))
ax.set_xticklabels(cars['car_names'], rotation=50,fontsize='medium')

# We're setting the title of the plot to describe its content. It's like 
# giving our artwork a title that captures its essence and emotional impact.
ax.set_title('Miles Per Gallon of Cars in mtcars')

# We're adding labels to the x-axis and y-axis to provide context for the 
# data being displayed. It's like adding captions to our artwork to guide 
# the viewer's interpretation.
ax.set_xlabel('car names')
ax.set_ylabel('miles/gal')

# We're setting the y-axis limits to focus on a specific range of values. 
# It's like zooming in on a particular section of our journey to explore 
# it in more detail.
ax.set_ylim([0,45])

# We're annotating a specific data point on the plot, highlighting its 
# significance. It's like adding a note to our artwork, providing additional 
# context for the viewer's understanding.
ax.annotate('Toyota Corolla', xy=(19,33.9), xytext=(21,35), arrowprops=dict(facecolor='black', shrink=0.05))

# We're adding a legend to the plot to provide additional information about 
# the displayed data. It's like providing a key to help the viewer interpret 
# the visual elements in our artwork.
ax.legend(loc='best')

# We're saving the plot as an image file for future reference or sharing. 
# It's like preserving our artwork in a tangible form, allowing others to 
# experience it even when they're not present.
plt.savefig('carPlot.jpg')

```

### Visualizing Time Series
```python

## Visualizing time series
# Yo, we're kicking things off with the essential tools 
# and modules. Just like gearing up before hitting 
# Super Walmart!

# Importing numpy for powerful numerical operations
import numpy as np

# Importing randn from numpy.random for generating random 
# numbers (think of it as grabbing those surprise deals 
# at Walmart!)
from numpy.random import randn

# Importing pandas for handling our data like a boss
import pandas as pd

# Importing Series and DataFrame from pandas for creating 
# and manipulating our data structures (imagine these as 
# our shopping carts for loading up on data goodies)
from pandas import Series, DataFrame

# Importing matplotlib.pyplot for creating awesome plots 
# and visualizations
import matplotlib.pyplot as plt

# Importing rcParams from pylab to customize plot parameters 
# (just like personalizing your shopping experience at Walmart)
from pylab import rcParams

# Importing seaborn for adding some extra style and 
# attractiveness to our plots (like finding that stylish 
# shirt at Walmart!)
import seaborn as sns

# Time to set the stage for our data visualizations, just 
# like arranging the displays at Super Walmart to catch 
# everyone's eye!

# Ensuring our plots appear inline in the notebook, making 
# it convenient for us to see the results right here.
%matplotlib inline

# Adjusting the figure size for our plots, just like 
# arranging the shelves to showcase our products in 
# Super Walmart.
rcParams['figure.figsize'] = 5, 4

# Setting the style for our plots, giving them a clean and 
# stylish look like the aisles in Super Walmart.
sns.set_style('whitegrid')

### The simplest time series plot
# Yo, we're about to load up our data, just like stocking 
# the shelves at Super Walmart!

# Defining the file path to our dataset, similar to 
# pinpointing the location of merchandise in the store.
address = '/workspaces/Python-for-Data-Science-and-Machine-Learning/Superstore-Sales.csv'

# Reading the CSV file into a DataFrame, as if we're 
# unpacking boxes of products at Super Walmart.
# Here, we're setting the 'Order Date' column as the index, 
# encoding the file using 'cp1252', and parsing dates to 
# ensure proper date handling.
store_df = pd.read_csv(address, index_col='Order Date', 
                        encoding='cp1252', parse_dates=True)

# Displaying the first few rows of our DataFrame, just 
# like checking out the front displays at Super Walmart to 
# see what's on offer.
store_df.head()

store_df['Order Quantity'].plot()
# Time to create a smaller sample of our dataset, like 
# selecting a few aisles to focus on at Super Walmart.

# Randomly selecting 100 rows from our DataFrame, just 
# like grabbing a mix of products from various sections 
# of Super Walmart.
store_df2 = store_df.sample(n=100, random_state=100, axis=0)

# Labeling the x-axis of our plot, similar to indicating 
# the time of day at Super Walmart.
plt.xlabel('Order Date')

# Labeling the y-axis of our plot, like specifying the 
# quantity of products sold at Super Walmart.
plt.ylabel('Order Quantity')

# Setting the title for our plot, akin to naming a sales 
# report at Super Walmart.
plt.title('Super Walmart Sales')

# Plotting the 'Order Quantity' column from our sample 
# DataFrame, resembling displaying sales data for a 
# particular product at Super Walmart.
store_df2['Order Quantity'].plot()



```

### Creating Statistical Data Graphics in Seaborn
```python

```






