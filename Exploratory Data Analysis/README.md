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