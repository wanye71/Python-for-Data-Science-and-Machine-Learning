# Python for Data Science and Machine Learning - Exploratory Data Analysis


## Table of Contents

1. [Simple arithmetic](#simple-arithmetic)
2. [Genrating summary statistics](#genrating-summary-statistics)
3. [Summarizing categorical data](#summarizing-categorical-data)
4. [Pearson correlation analysis](#pearson-correlation-analysis)
5. [Spearman rank correlation and Chi-square](#spearman-rank-correlation-and-chi-square)
6. [Extreme value analysis for outliers](#extreme-value-analysis-for-outliers)
7. [Multivariate analysis for outliers](#multivariate-analysis-for-outliers)


### Simple arithmetic
```python

# Creating a NumPy array representing daily stock prices 
# for a particular stock.
a = np.array([1,2,3,4,5,6])  # Daily stock prices
a
# Creating a 2D NumPy array representing stock prices for 
# multiple stocks over a period of time.
b = np.array([[10, 20, 30], [40, 50, 60]])  # Stock prices 
# for multiple stocks
b
### Creating arrays via assignment
# Setting the random seed to ensure reproducibility of random 
# number generation, similar to the concept of fixing a 
# starting point for stock price simulations.
np.random.seed(25)

# Generating random stock price data using normal distribution 
# with mean 0 and standard deviation 1, then scaling it by 36 to 
# represent stock prices.
c = 36 * np.random.randn(6)  # Random stock price data
c
# Creating an array representing the days of the month, similar 
# to a calendar month where each number corresponds to a day.
d = np.arange(1, 35)  # Array representing days of the month
d

### Multiplying matrices and basic linear algebra
# Multiplying each element of array 'a' by 10, akin to 
# increasing the stock prices by a factor of 10.
a * 10

# Adding each element of array 'c' to the corresponding 
# element of array 'a', mirroring the effect of combining 
# the stock prices from array 'c' with other factors, like 
# dividends, represented by array 'a'.
c + a

# Subtracting each element of array 'a' from the corresponding 
# element of array 'c', akin to analyzing the difference in 
# stock prices after a certain event represented by array 'a'.
c - a

# Multiplying each element of array 'c' with the corresponding 
# element of array 'a', analogous to calculating the product of 
# stock prices and quantities traded.
c * a

# Dividing each element of array 'c' by the corresponding 
# element of array 'a', analogous to calculating the ratio of 
# stock prices to dividends.
c / a

```

### Genrating summary statistics
```python
# Yo, check it! We're gearing up to analyze some volcanic 
# data, like diving into the urban jungle with our crew.

# Rolling with NumPy, our heavy-duty toolkit for crunching 
# numbers, like the muscle car revving up for action.
import numpy as np

# Here comes Pandas, the solid foundation for our data game, 
# making sure our stats stay on track, just like a street 
# racer's chassis holding it down.
import pandas as pd

# We're grabbing Series and DataFrame from the Pandas posse, 
# those slick data structures keeping our volcanic info 
# organized, like the street signs guiding the way.
from pandas import Series, DataFrame

# Scipy's on deck, packing serious statistical firepower 
# for analyzing volcanic activity, like the high-tech gear 
# helping us navigate the concrete jungle.
import scipy

# We're rolling with the stats crew from Scipy, hooking us 
# up with all the math magic we need to make sense of our 
# volcanic data, just like the expert advice from the 
# neighborhood elders.
from scipy import stats

# Setting the coordinates for the data eruption point
address = '/workspaces/Python-for-Data-Science-and-Machine-Learning/volcanos.csv'

# Activating the data eruption point and capturing the volcanic
# activity into the 'volcanos' DataFrame
volcanos = pd.read_csv(address)

# Observing the initial eruption from the dataset to get a
# glimpse of the volcanic landscape
volcanos.head()

# Renaming the features to better interpret the volcanic
# eruption data and understand the terrain
volcanos.columns = ['VolcanoID', 'V_Name', 'Country', 'Region',
                    'Subregion', 'Latitude', 'Longitude', 'PEI',
                    'H_active', 'VEI_Holoce', 'hazard', 'class',
                    'risk']

# Observing the modified volcanic landscape to see how the
# eruption features have been renamed
volcanos.head()

### Looking at summary statistics that decribe a variable's numeric values
# volcanos.sum()

# # TypeError: can only concatenate str (not "int") to str
# In the realm of molten fire and earth's command,
# We seek the numeric might, firm and grand.
# Selecting types of data, numeric and bold,
# With volcanic power, our sums unfold.

# Selecting columns with numeric data types, excluding
# any non-numeric columns, to perform summation.
# The `include` parameter specifies the data types to 
# include in the selection, here set to only include 
# numeric types using `np.number`.
# The `.sum()` method calculates the sum along the rows 
# (axis 0 by default) of the selected numeric columns.
volcanos.select_dtypes(include=[np.number]).sum()

# Beneath the volcano's fiery gaze,
# We seek the numbers that amaze.
# Selecting data types of numeric might,
# To sum them up, our goal in sight.

# Selecting columns with numeric data types, excluding
# any non-numeric columns, to perform summation.
# Here, 'number' is used as shorthand for numeric data types.
# The `.sum()` method calculates the sum along the rows 
# (axis 0 by default) of the selected numeric columns.
volcanos.select_dtypes('number').sum()

# Amidst the molten lava's glow,
# We tally numbers row by row.
# Summing up the volcanic might,
# With only numeric columns in our sight.

# Summing up the numeric values across all columns,
# excluding non-numeric ones, like an eruption's fiery flow.
# The `numeric_only=True` parameter ensures that only 
# numeric columns are considered for summation.
volcanos.sum(numeric_only=True)

# Like shedding a layer of volcanic crust,
# We drop the 'VolcanoID' column, leaving behind the lust.
# Summing up the remaining volcanic might,
# With only numeric columns in our sight.

# Dropping the 'VolcanoID' column along the specified axis,
# then summing up the numeric values across all remaining columns.

# Here, we exclude the 'VolcanoID' column from our dataset 
# using the drop() method, specifying the axis along which 
# to drop the column. Then, we sum the numeric values across 
# all remaining columns using the sum() method with the 
# numeric_only parameter set to True.
volcanos.drop('VolcanoID', axis=1).sum(numeric_only=True)

# As the volcanos rumble and roar, 
# Let's sum up their data to explore.

# The `sum` function is used to calculate the sum of values
# along the specified axis, considering only numeric columns.
# Here, we sum across each row, treating only numeric values,
# to get the total sum for each volcano.

# Parameters:
# - axis: Specifies the axis along which the sum is computed.
#   - axis=1: Summation is performed along rows.
# - numeric_only: If True, only numeric columns will be summed,
#   excluding non-numeric columns like object or categorical.
#   Defaults to True.
# Returns:
# - Series: The sum of values for each volcano.
volcanos_sum = volcanos.sum(axis=1, numeric_only=True)

# Amidst the magma and rock, seeking the median, 
# our quest on the volcanic terrain is steadily leadin'.

# Calling `median()` upon our dataset of fire, 
# we summon forth the middle value, a measure we admire.

# With `numeric_only=True`, we specify our interest, 
# focusing solely on numerical columns, a directive 
# we insist.

# Parameters:
# - numeric_only: A boolean parameter indicating whether to consider only 
#   numeric data when calculating the median. If set to True, non-numeric 
#   columns will be excluded from the calculation. Defaults to False.
volcanos.median(numeric_only=True)

# With volcanoes in mind, dropping the 'VolcanoID' column 
# to attain the mean is quite divine.

# The `drop()` function, a powerful tool, removes the 
# specified column axis-wise, keeping our data cool. 
# Setting `axis=1` ensures we target columns, not rows, 
# leaving behind only the info that glows.

# The `mean()` function, post-drop, calculates the average 
# of the remaining data, a move that's not a flop. By 
# specifying `numeric_only=True`, we ensure that only 
# numeric columns are included in our review, allowing us 
# to analyze and construe.

volcanos.drop('VolcanoID', axis=1).mean(numeric_only=True)

# In the land of fire and ash, where volcanoes reign supreme,
# We drop the ID of molten rocks, a dream within a dream.
# Seeking the mightiest value, the highest peak we chase,
# Numeric only, to exclude the rest, in this volcanic race.

# Parameters:
#   - axis: Specifies the axis along which to drop the column. 
#           0 for rows and 1 for columns.
#   - numeric_only: If True, only include numeric data in the calculation 
#                    of the maximum value. If False, include all data types.
# The max() function returns the maximum value along the specified axis, 
# excluding non-numeric columns if numeric_only is set to True.
volcanos.drop('VolcanoID', axis=1).max(numeric_only=True)

### Looking at summary statistics that describe variable distribution
# In the land of fire and ash, where volcanoes cast their shadow,
# We drop the mighty ID, from the depths, we let it go.
# Standard deviation we seek, a measure of dispersion's might,
# Numeric only, to exclude all but the numeric, shining bright.

# Parameters:
#   - axis: Specifies the axis along which to drop the column. 
#           0 for rows and 1 for columns.
#   - numeric_only: If True, only include numeric data in the calculation 
#                    of the standard deviation. If False, include all data types.
# The std() function calculates the standard deviation of the data along the specified axis.
# Standard deviation measures the dispersion of values in a dataset from the mean. 
# It is calculated as the square root of the variance.
volcanos.drop('VolcanoID', axis=1).std(numeric_only=True)

# In a land of molten rock, where volcanoes reign supreme,
# We drop the specified column, like a boulder in a stream.
# Variance we seek, a measure of spread and range,
# Numeric only, to exclude non-numeric, it's no longer strange.

# Parameters:
#   - axis: Specifies the axis along which to drop the column. 
#           0 for rows and 1 for columns.
#   - numeric_only: If True, only include numeric data in the calculation 
#                    of the variance. If False, include all data types.
# The var() function calculates the variance of the data along the specified axis.
# Variance is a measure of how much the values in a dataset vary from the mean. 
# It is calculated as the average of the squared differences from the mean.
volcanos.drop('VolcanoID', axis=1).var(numeric_only=True)

# A volcano awakens, its fury held tight,
# We capture its rumble, with all of our might.

# Assigning the 'H_active' column to 'h_active',
# Like a seismic recorder, our data we'll derive.
h_active = volcanos.H_active

# Counting the eruptions, their frequency we track,
# To understand their patterns, and how they stack.

# Using the value_counts() function, it's key,
# As it tallies the eruptions, for all to see.
# This function computes the frequency of unique values in a Series.
h_active.value_counts()

# In the realm of fire, where volcanoes dwell,
# Let's explore their secrets, tales they tell.

# With describe() in hand, we step into the glow,
# Unveiling the volcano's story, row by row.

# This function whispers of eruptions past,
# Sketching the landscape of our fiery blast.
volcanos.describe()
```

### Summarizing categorical data
```python
## Summarizing categorical data using pandas

import numpy as np
import pandas as pd

from pandas import Series, DataFrame

### The basics

# The {address} variable holds the file path to the storm data CSV, 
# allowing us to access and load the data into our analysis 
# environment. It serves as the starting point for our exploration 
# into the stormy seas of data analysis.
address = '../storms_points.csv'

# Upon loading the storm data from the provided CSV file, the 
# {storms} variable becomes a container for our dataset, holding 
# information about various storm events, including their names, 
# locations, and intensity.
storms = pd.read_csv(address)

# After loading the data, the column names of the {storms} DataFrame 
# are renamed for clarity and consistency, ensuring that each column 
# accurately represents the corresponding aspect of storm data.
storms.columns = ['storm_name', 'year', 'month', 'day', 'hour', 'minute', 
                  'timestamp', 'record_ident', 'status', 'latitude', 
                  'longitude', 'max_wind_kts', 'max_wind_kph', 
                  'max_wind', 'min_press']

# Setting the storm names as the index of the {storms} DataFrame 
# provides a convenient way to identify and reference specific 
# storm events during our analysis, acting as a navigational guide 
# through the dataset.
storms.index = storms.storm_name

# Displaying the first 10 rows of the storm data, the {storms.head()} 
# function offers a glimpse into the structure and contents of the 
# DataFrame, showcasing key information such as storm names, dates, 
# and wind speeds.
storms.head(10)


# As we sail through the stormy seas of data, let's harness the power 
# of the winds with the {max_wind} variable, capturing the maximum wind 
# speed recorded for each storm event. Like a fierce gust of wind, it 
# sweeps through the dataset, revealing the intensity of each storm's 
# fury.

# By counting the occurrences of different wind speeds using the 
# {value_counts()} method, we gain insights into the distribution of 
# maximum wind speeds across the storm dataset, akin to charting the 
# varying strengths of tropical tempests in our data exploration journey.
max_wind = storms.max_wind
max_wind.value_counts()


# Amidst the storm of data, we seek to extract specific coordinates 
# and temporal information to guide our journey through the turbulent 
# seas.

# Extracting the categorical storm data related to latitude, longitude, 
# and year, we create a new dataframe named {storms_cat} by selecting 
# columns 'latitude', 'longitude', and 'year' from the {storms} 
# dataframe.
storms_cat = storms[['latitude', 'longitude', 'year']]

# Displaying the first few rows of the {storms_cat} dataframe to provide 
# a glimpse into the extracted categorical storm data, offering insight 
# into the initial coordinates and temporal details of our stormy 
# expedition.
storms_cat.head()


# In the storm's domain we dwell,
# Collecting data, our story to tell,
# With latitude and longitude in our hand,
# And the year, to understand.

# Grouped by year, we explore,
# Unveiling trends, seeking more,
# Descriptive stats, our guide,
# Revealing patterns far and wide.

# Grouping the storm data by 'year'
year_group = storms_cat.groupby('year')

# Displaying descriptive statistics for the grouped data
year_group.describe()


### Transforming variables to categorical data type

# Amidst the stormy haze, a new label we raise,
# Assigning each year a categorical blaze,
# With dtype='category', we define its form,
# A unique identifier to weather the storm.

# Creating a new column 'group' in the storms DataFrame,
# Setting its values to the 'year' column,
# And specifying the data type as 'category'
storms['group'] = pd.Series(storms.year, dtype='category')


# Retrieve the data type of the 'group' column,
# Ensuring consistency, as we need to control,
# This detail helps understand the data's form,
# Clear insights emerge, analysis can perform.
storms['group'].dtypes


# Count occurrences of each category in the 'group' column,
# With this insight, our analysis can enthrall,
# Knowing the distribution, we can plan ahead,
# Insights from data, our decisions are led.
storms['group'].value_counts()


### Describing categorical data with crosstabs

# Generate a cross-tabulation between 'year' and 'month' columns,
# Revealing insights, patterns that will enthral,
# Rows for years, columns for months, a matrix unfolds,
# Each cell a count, a story it holds.
pd.crosstab(storms['year'], storms['month'])

```

### Pearson correlation analysis
```python
# Starting with parametric methods in pandas and scipy

# Pandas is imported as 'pd', the go-to tool for data handling,
# NumPy joins in too, arrays and matrices commanding.

# Matplotlib and Seaborn, charting pals in the mix,
# Plotting stories, revealing trends, with visual tricks.

# RCParams, a behind-the-scenes wizard's spell,
# Crafting plot sizes, an art, I can tell.

# Lastly, Scipy arrives, with stats on its mind,
# Pearson correlation, a measure, it's designed.
import pandas as pd  # The data powerhouse, pandas
import numpy as np   # NumPy, arrays and matrices grand

import matplotlib.pyplot as plt  # Matplotlib's plotting thrill
import seaborn as sns             # Seaborn, charting skill
from pylab import rcParams        # RCParams, the plot size's command

import scipy                        # Scipy for stats and more
from scipy.stats import pearsonr   # Pearson correlation, to explore


# In the realm of visualization, where insights align, `%matplotlib
# inline` ensures plots are inline, just fine. With this magic command,
# plots appear in line, making visualization a breeze, a technique so
# divine.
%matplotlib inline

#***********************************************************#

# rcParams sets the dimensions, a canvas to define, `figure.figsize`
# determines width and height, a size so prime. With dimensions
# adjusted, plots come alive, in size and proportion, they strive.
# rcParams['figure.figsize'] = 12, 5

#***********************************************************#

# With the style of the plot, a canvas so neat, `whitegrid` is set,
# making plots complete. A backdrop of white, with gridlines so light,
# seaborn sets the stage, for data's insight.
sns.set_style("whitegrid")


#         ### Obese = BMI greater than or equal to 30
#         ### Overweight = BMI greater than or equal to 25 but less than 30


# In the world of data, where insights unfold, a file is sought, its
# tales untold. 'address' is the path, where data resides, to be
# explored and analyzed, where knowledge abides.
address = '../obesity_data.csv'

#***********************************************************#

# From the depths of files, 'obese_data' arises, a DataFrame formed, its
# structure comprises. With 'read_csv', it ingests the data, a table
# it forms, where numbers await, amidst digital storms.
obese = pd.read_csv(address)

#***********************************************************#

# A transformation unfolds, columns evolve, 'obese' adapts, its
# structure resolves. 'Age', 'Gender', and 'Height', familiar faces
# remain, joined by 'Weight', 'BMI', and 'PhysicalActivityLevel', a
# comprehensive domain. Yet another addition, 'ObesityCategory', it
# joins the fold, completing the ensemble, a story yet untold.

# The 'columns' attribute of the 'obese' DataFrame is assigned a new list
# containing the column names 'Age', 'Gender', 'Height', 'Weight', 'BMI',
# 'PhysicalActivityLevel', and 'ObesityCategory', thereby renaming the
# columns accordingly.
obese.columns = ['Age', 'Gender', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel', 'ObesityCategory']


#***********************************************************#
# # A glimpse we take, a peek so fine, 'obese_cols' reveals, its first
# # ten in line. The head is shown, a preview complete, of data
# # transformed, now neat and discreet.
# obese.head(10)
obese.head(5)

# obese.dtypes

# In the realm of visualization, where patterns arise, 'sns.pairplot'
# takes its guise. With 'obese' in hand, it creates a plot, pairs
# abound, revealing relationships sought.

# 'sns.pairplot' generates a grid of pairwise plots for the numerical
# columns of the 'obese' DataFrame, offering a glimpse into the data's
# distribution and correlations.
sns.pairplot(obese)


# In the realm of selection, where variables align, 'x' is crafted,
# where choices entwine. From 'obese', it plucks, 'BMI', 'Weight', and
# 'Gender', a trio so fine.

# 'x' is created as a DataFrame containing the selected columns 'BMI',
# 'Weight', and 'Gender' from the 'obese' DataFrame, forming a subset
# tailored for analysis.
x = obese[['BMI', 'Weight', 'Gender']]

#***********************************************************#

# In the realm of visualization, where insights unfold, 'sns.pairplot'
# takes its toll. With 'x' in hand, it paints a plot, pairs galore,
# revealing relationships, and more.

# 'sns.pairplot' generates a grid of pairwise plots for the selected
# variables 'BMI', 'Weight', and 'Gender' from the 'x' DataFrame,
# providing a visual exploration of their relationships.
sns.pairplot(x, vars=['BMI', 'Weight', 'Gender'])



### Using scipy to calculate the Pearson correlation coefficient

# From the depths of 'obese', 'Weight' emerges, a series so bold, its
# values urge. An array of weights, where pounds abide, a variable
# measured, with physicality tied.
weight = obese['Weight']

#***********************************************************#

# Another contender, 'PhysicalActivityLevel' strides, a measure of
# movement, where activity resides. From 'obese' it comes, a column so
# keen, activity levels gauged, in every scene.
activity = obese['PhysicalActivityLevel']

#***********************************************************#

# A third player enters, 'BMI' is here, a metric of health, drawing
# near. From 'obese' it's drawn, a calculation so fine, a ratio of
# weight and height, in every line.
bmi = obese['BMI']

#***********************************************************#

# The stage is set, the players aligned, 'pearsonr' takes flight, its
# purpose defined. With 'weight' and 'activity', it seeks a bond, a
# correlation revealed, a truth so fond.

# The Pearson correlation coefficient and p-value are computed for the
# 'weight' and 'activity' variables using the 'pearsonr' function.
pearsonr_coefficient, p_value = pearsonr(weight, activity)

#***********************************************************#

# The result is revealed, the coefficient displayed, a measure of
# correlation, where insight's conveyed.
print('Pearson Correlation Coefficient %0.3f' % (pearsonr_coefficient))


# From the depths of 'obese', 'Weight' emerges, a series so bold, its
# values urge. An array of weights, where pounds abide, a variable
# measured, with physicality tied.
weight = obese['Weight']

#***********************************************************#

# A third player enters, 'BMI' is here, a metric of health, drawing
# near. From 'obese' it's drawn, a calculation so fine, a ratio of
# weight and height, in every line.
bmi = obese['BMI']

#***********************************************************#

# The stage is set, the players aligned, 'pearsonr' takes flight, its
# purpose defined. With 'weight' and 'bmi', it seeks a bond, a
# correlation revealed, a truth so fond.

# The Pearson correlation coefficient and p-value are computed for the
# 'weight' and 'bmi' variables using the 'pearsonr' function.
pearsonr_coefficient, p_value = pearsonr(weight, bmi)

#***********************************************************#

# The result is revealed, the coefficient displayed, a measure of
# correlation, where insight's conveyed.
print('Pearson Correlation Coefficient %0.3f' % (pearsonr_coefficient))


# A third player enters, 'BMI' is here, a metric of health, drawing
# near. From 'obese' it's drawn, a calculation so fine, a ratio of
# weight and height, in every line.
bmi = obese['BMI']

#***********************************************************#

# Another contender, 'PhysicalActivityLevel' strides, a measure of
# movement, where activity resides. From 'obese' it comes, a column so
# keen, activity levels gauged, in every scene.
activity = obese['PhysicalActivityLevel']

#***********************************************************#

# The stage is set, the players aligned, 'pearsonr' takes flight, its
# purpose defined. With 'bmi' and 'activity', it seeks a bond, a
# correlation revealed, a truth so fond.

# The Pearson correlation coefficient and p-value are computed for the
# 'bmi' and 'activity' variables using the 'pearsonr' function.
pearsonr_coefficient, p_value = pearsonr(bmi, activity)

#***********************************************************#

# The result is revealed, the coefficient displayed, a measure of
# correlation, where insight's conveyed.
print('Pearson Correlation Coefficient %0.3f' % (pearsonr_coefficient))


# In the realm of selection, where variables align, 'x' is crafted,
# where choices entwine. From 'obese', it plucks, 'BMI', 'Weight', and
# 'Height', a trio so fine.

# 'x' is created as a DataFrame containing the selected columns 'BMI',
# 'Weight', and 'Height' from the 'obese' DataFrame, forming a subset
# tailored for analysis.
x = obese[['BMI', 'Weight', 'Height']]

#***********************************************************#

# The stage is set, the trio in hand, 'corr' is calculated, a matrix so
# grand. Correlations abound, between each pair, revealing relationships,
# beyond compare.

# The correlation matrix is computed for the columns of the 'x' DataFrame,
# providing insights into the relationships between 'BMI', 'Weight', and
# 'Height'.
corr = x.corr()

#***********************************************************#

# The result is revealed, a matrix displayed, where correlations reside,
# relationships unveiled.
corr


### Using Seaborn to visualize the Pearson correlation coefficient

# In the realm of visualization, where insights take flight, 'sns.heatmap'
# casts its light. With 'corr' in hand, a matrix of correlation, it
# paints a picture, a visualization sensation.

# 'sns.heatmap' generates a heatmap visualization of the correlation matrix
# 'corr', with column names used as tick labels on both the x-axis and
# y-axis to denote the variables being compared.
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
```

### Spearman rank correlation and Chi-square
```python

```

### Extreme value analysis for outliers
```python

```

### Multivariate analysis for outliers
```python

```