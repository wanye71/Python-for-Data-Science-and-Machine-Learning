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


```

### Pearson correlation analysis
```python

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