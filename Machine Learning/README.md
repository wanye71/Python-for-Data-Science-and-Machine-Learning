
# Python for Data Science and Machine Learning - Machine Learning

## Table of Contents

1. [Cleaning and treating categorical variables](#cleaning-and-treating-categorical-variables)
2. [Transforming dataset distribution](#transforming-dataset-distribution)
3. [Applied machine learning](#applied-machine-learning)

### Cleaning and treating categorical variables
```python
# Cleaning and treating categorical variables


# In the bustling city of data, where insights 
# thrive, NumPy steps forth, like a robust archive.
# With arrays akin to people, each with its role,
# It orchestrates the symphony, of data's stroll.

# Import NumPy for numerical computing, akin to a 
# bustling city where data insights thrive.
import numpy as np

#**************************************************#
# In the community of data, where stories are 
# told, Pandas stands tall, like a leader bold.
# With DataFrames like people, each with a tale,
# It curates the narrative, without fail.

# Import Pandas for data manipulation, akin to a 
# vibrant community where stories are told.
import pandas as pd

#**************************************************#
# Import DataFrame from Pandas for convenience, 
# symbolizing individuals within the community.
from pandas import DataFrame

#**************************************************#
# In the landscape of knowledge, where wisdom 
# gleams, Scikit-learn emerges, like a mentor's 
# dreams. With encoders like guides, and 
# transformations like paths, It guides the journey,
# through learning's drafts.


# Import LabelEncoder and OneHotEncoder from 
# Scikit-learn for preprocessing, resembling 
# mentors guiding the journey of learning.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


#**************************************************#

# In the realm of information, where stories unfold,
# A dictionary named 'data' with tales to behold.
# Names, ages, and genders, each with their role,
# Forming a narrative, of people's stroll.
#**************************************************#

# Define a dictionary 'data' containing information about individuals, akin to stories in a book.
data = {
    'names': ['steve', 'john', 'richard', 'sarah', 
              'randy', 'micheal', 'julie'],
    'age': [20, 22, 20, 21, 24, 23, 22],
    'gender': ['Male', 'Male', np.nan, 'Female',
               np.nan, 'Male', np.nan],
    'rank': [2, 1, 4, 5, 3, 7, 6]
}

#**************************************************#

# In the realm of organization, where data takes form,
# A DataFrame named 'df', like a shelter from the storm.
# With rows and columns, each with a role to play,
# Crafting a tableau, of data's display.
#**************************************************#

# Create a DataFrame 'df' from the dictionary 'data', symbolizing an organized tableau of information.
df = DataFrame(data)

#**************************************************#

# Display the DataFrame 'df', a snapshot in time,
# With stories of people, in their prime.
df


# In the landscape of transformation, where changes unfurl,
# A DataFrame named 'df' undergoes a whirl.
# Dropping a column named 'gender', with a purpose to thrive,
# Simplifying the narrative, as tales revive.


# Drop the 'gender' column from the DataFrame 'df', akin to removing a character from a story.
df = df.drop('gender', axis=1)

#**************************************************#
# Display the modified DataFrame 'df', a narrative refined,
# With characters departed, their stories defined.
df


## Label Encoding

### Step 1

# In the realm of encoding, where labels find their place,
# A LabelEncoder named 'label_encoder' begins its race.
# Training on the 'names' column, a task to fulfill,
# Mapping characters to numbers, with precision and skill.


# Create a LabelEncoder named 'label_encoder', akin to assigning roles to characters in a play.
label_encoder = LabelEncoder()

#**************************************************#
# Fit the LabelEncoder 'label_encoder' to the 'names' column of the DataFrame 'df',
# akin to training it on the characters' names in the narrative.
label_encoder.fit(df['names'])



### Step 2

# In the world of transformation, where characters evolve,
# The LabelEncoder 'label_encoder' resolves.
# Transforming the 'names' column into encoded form,
# Characters become numbers, a novel reform.


# Transform the 'names' column of the DataFrame 'df' into encoded form using the trained LabelEncoder,
# akin to assigning numeric identities to characters in the narrative.
label_encoded_names = label_encoder.transform(df['names'])

#**************************************************#
# Display the encoded names, a transformation complete,
# Characters now numbered, their identities discreet.
label_encoded_names


## One Hot Encoder

# In the realm of encoding, where diversity thrives,
# OneHotEncoder 'onehot_encoder' arrives.
# With sparse_output set to False, a choice to abide,
# Dense arrays emerge, from the encoding tide.


# Create a OneHotEncoder named 'onehot_encoder', akin to expanding the narrative with diverse elements.
onehot_encoder = OneHotEncoder(sparse_output=False)



# In the realm of encoding, where patterns take flight,
# OneHotEncoder 'onehot_encoder' shines bright.
# Fitting on the 'names' column, a task to fulfill,
# Capturing the essence, each unique element's skill.


# Fit the OneHotEncoder 'onehot_encoder' to the 'names' column of the DataFrame 'df',
# akin to learning the unique patterns of characters' names in the narrative.
onehot_encoder.fit(df[['names']])



# In the realm of transformation, where diversity gleams,
# OneHotEncoder 'onehot_encoder' brings dreams.
# Transforming the 'names' column into one-hot form,
# Each character's essence, like a beacon to swarm.


# Transform the 'names' column of the DataFrame 'df' into one-hot encoded form using the fitted OneHotEncoder,
# akin to representing each character's presence as a binary vector.
onehot_encoded_names = onehot_encoder.transform(df[['names']])



# In the world of creation, where data takes shape,
# DataFrame 'onehot_encoded_df' finds its escape.
# With encoded names as columns, a tableau refined,
# Each character's essence, now clearly defined.


# Create a DataFrame 'onehot_encoded_df' from the one-hot encoded names,
# akin to crafting a tableau of characters' essence.
onehot_encoded_df = DataFrame(onehot_encoded_names, columns=onehot_encoder.categories_)

#**************************************************#
# Assign the original 'names' column from the DataFrame 'df' to 'onehot_encoded_df',
# preserving the characters' identities within the tableau.
onehot_encoded_df['names'] = df[['names']]

#**************************************************#
# Display the DataFrame 'onehot_encoded_df', a narrative complete,
# Characters encoded, their essence concrete.
onehot_encoded_df
```

### Transforming dataset distribution
```python
# Transforming Dataset Distributions


# In the landscape of data, where insights take flight,
# Pandas tends the fields, with its data might.
# A toolbox of data structures and functions, it brings,
# For data manipulation and analysis, it sings.


# Importing pandas for data manipulation and analysis.
import pandas as pd

#**************************************************#
# Amidst the arrays' realm, where numbers align,
# NumPy stands tall, with its numerical shine.
# Arrays and matrices, its domain so vast,
# For numerical computations, unsurpassed.

# Importing NumPy for numerical computations with arrays.
import numpy as np

#**************************************************#
# On the canvas of visualization, where stories ignite,
# Matplotlib paints the picture, colors so right.
# A canvas for plots, clear and bright,
# Data speaks volumes, without any fright.

# Importing Matplotlib for visualization of data.
import matplotlib.pyplot as plt

#**************************************************#
# In the realm of transformation, where scales align,
# MinMaxScaler and scale, their powers combine.
# With features rescaled, a new horizon to define,
# Normalization and standardization, in data's design.

# Importing MinMaxScaler and scale for feature scaling.
from sklearn.preprocessing import MinMaxScaler, scale


# In the world of data, where cars roam free,
# DataFrame 'cars_data' emerges, you see.
# From a CSV file, its tales unfurl,
# A tableau of vehicles, ready to swirl.

# The file path to the CSV containing used cars data.
address = '../used_cars.csv'

#**************************************************#
# In the realm of data, where files converge,
# DataFrame 'cars_data' is forged, with a surge.
# From the CSV file, its journey begins,
# Loaded into memory, where the story spins.

# Read the CSV file into a DataFrame 'cars_data',
# unleashing the tales of used cars onto the screen.
cars_data = pd.read_csv(address)

#**************************************************#
# Display the first few rows of DataFrame 'cars_data',
# A glimpse into the world of used cars, ready to invade.
cars_data.head()


# In the canvas of visualization, where data paints a tale,
# Matplotlib takes the lead, without fail.
# A line plot emerges, with mileage in sight,
# The journey of each car, in the day and the night.

# Create a line plot of 'Mileage' from DataFrame 'cars_data',
# depicting the mileage of each car over time.
plt.plot(cars_data['Mileage'])


## Normalization

# In the realm of scaling, where features align,
# MinMaxScaler emerges, its powers combine.
# With data transformed, within a range so right,
# Features rescaled, in data's light.

# Create an instance of MinMaxScaler to scale the 'Mileage' feature,
# ensuring that data falls within a specified range.
minmax_scaler = MinMaxScaler()

#**************************************************#
# In the realm of transformation, where scales take flight,
# MinMaxScaler fits the data, with its might.
# Ensuring the 'Mileage' feature, rescaled in its span,
# A range defined, for each car's journey to stand.

# Fit the MinMaxScaler to the 'Mileage' feature in DataFrame 'cars_data',
# determining the scaling factors to transform the data.
minmax_scaler.fit(cars_data[['Mileage']])


# In the realm of transformation, where scales take flight,
# Features are rescaled, within a range so right.
# MinMaxScaler applies its magic, with precision so fine,
# Transforming the 'Mileage' feature, in data's design.

# Transform the 'Mileage' feature in DataFrame 'cars_data'
# using the fitted MinMaxScaler, ensuring data falls within the specified range.
scaled_data = minmax_scaler.transform(cars_data[['Mileage']])


# In the canvas of visualization, where data takes flight,
# Matplotlib paints the picture, with colors so right.
# A line plot emerges, with scaled data in sight,
# The journey of each car, in a normalized light.

# Create a line plot of the scaled 'Mileage' data,
# depicting the transformed mileage of each car over time.
plt.plot(scaled_data)


## Standardization

# In the realm of transformation, where scales align,
# StandardScaler emerges, its powers combine.
# With features standardized, a new horizon to define,
# Normalization achieved, in data's design.

# Standardize the 'Mileage' feature in DataFrame 'cars_data'
# using the scale function, ensuring mean of 0 and standard deviation of 1.
standard_scalar = scale(cars_data[['Mileage']])

#**************************************************#
# In the canvas of visualization, where data takes flight,
# Matplotlib paints the picture, colors so right.
# A line plot emerges, with standardized data in sight,
# The journey of each car, in a standardized light.

# Create a line plot of the standardized 'Mileage' data,
# depicting the standardized mileage of each car over time.
plt.plot(standard_scalar)
```

### Applied machine learning
```python
# Applied machine learning: Starter problem

# In the garden of data, where insights bloom bright,
# Pandas tends the fields, with its data might.
# DataFrame 'pd' emerges, with data so fine,
# Ready to explore, each row, each line.

import pandas as pd

#**************************************************#
# Amidst the garden's blooms, where divisions take flight,
# train_test_split comes forth, its wisdom to invite.
# Splitting the garden's treasures, between train and test,
# A step towards learning, with results the best.

from sklearn.model_selection import train_test_split

#**************************************************#
# In the realm of flowers, where decisions are set,
# DecisionTreeClassifier takes the bet.
# Predictions it makes, with nodes so bright,
# A model to guide, through data's flight.

from sklearn.tree import DecisionTreeClassifier

#**************************************************#
# In the realm of evaluation, where metrics shine,
# Metrics from sklearn, with numbers so fine.
# Accuracy, precision, recall, and more,
# Evaluation of models, to reveal the score.

from sklearn import metrics


# In the garden of data, where insights bloom bright,
# Pandas tends the fields, with its data might.
# DataFrame 'pd' emerges, with data so fine,
# Ready to explore, each row, each line.

import pandas as pd

#**************************************************#
# Amidst the garden's blooms, where treasures take flight,
# Iris data is fetched, a dataset so right.
# From CSV it arises, with columns so clear,
# Ready for analysis, without any fear.

#**************************************************#
# Define the relative file path to the Iris dataset CSV file.
# The '..' indicates the parent directory of the current working directory.
# 'iris.csv' is the filename of the dataset.
address = '../iris.csv'

#**************************************************#
# Use the pandas library's read_csv function to load the Iris dataset from the CSV file.
# The result is stored in a DataFrame object, which is a 2-dimensional labeled data structure.
# DataFrames are similar to SQL tables or Excel spreadsheets.
import pandas as pd
iris_data = pd.read_csv(address)

#**************************************************#
# Display the first five rows of the DataFrame using the head() method.
# This is useful for quickly testing if your object has the right type of data in it.
# By default, head() displays the first five rows of the DataFrame.
iris_data.head()


# In the garden of data, where insights bloom bright,
# Pandas tends the fields, with its data might.
# DataFrame 'pd' emerges, with data so fine,
# Ready to explore, each row, each line.

# Use the unique() method on the 'Species' column of the iris_data DataFrame.
# This method identifies all unique values in the column, which in the case of the Iris dataset,
# will typically be the different species of Iris flowers.
# The result is an array of unique species names, providing a quick way to see all the distinct species present in the dataset.
iris_data.Species.unique()


## Separating features and lables

# In the data's garden, where features take flight,
# A slice of the DataFrame, to bring insight.
# Column selection, precise and keen,
# x holds the features, the data's scene.

# The 'iloc' method is used for position-based indexing in pandas
# DataFrames. Here, it's selecting all rows (indicated by ':') and the
# columns from index 1 to 4 (indicated by '1:5'). The end index in a
# range is exclusive, so '1:5' selects columns at index 1, 2, 3, and
# 4. This will typically correspond to the 'SepalWidth', 'PetalLength',
# 'PetalWidth', and another column if present. The result is assigned
# to the variable 'x', creating a new DataFrame with just the selected
# columns.

x = iris_data.iloc[:, 1:5]

#**************************************************#
# Display the contents of the variable 'x'. Since 'x' is a DataFrame,
# the output will be formatted as a table where each row represents an
# observation and each column represents a feature of the Iris dataset.
# This is useful for verifying the data contained in 'x' and ensuring
# that the DataFrame has been correctly created.
x



# In the garden of data, where blooms inspire, iris_data awaits, with
# its data attire. Rows and columns, a garden so vast, where insights
# flourish, the die is cast.

# Use the 'iloc' method to select all rows and the column at index 5.
# This typically corresponds to the 'Species' column in the Iris
# dataset, which contains the target variable (the species of the Iris
# flower). The result is assigned to the variable 'y', creating a
# Series containing the target values.

y = iris_data.iloc[:, 5]

#**************************************************#
# Display the contents of the variable 'y'. Since 'y' is a Series,
# the output will be formatted as a single column where each row
# represents the species of an Iris flower in the dataset. This is
# useful for verifying the data contained in 'y' and ensuring that the
# Series has been correctly created.
y


## Train Test Split

# Split the dataset into training and testing sets using the
# train_test_split function from scikit-learn.
# The 'x' DataFrame contains the features, and the 'y' Series
# contains the target variable.

# The test_size parameter specifies the proportion of the dataset to
# include in the test split. Here, it's set to 0.3, indicating that
# 30% of the data will be used for testing.

# The random_state parameter controls the shuffling of the dataset
# before splitting. Setting it to 0 ensures reproducibility of the
# results.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


## Training Decision Tree Classifier

# A classifier from Decision Tree we spawn,
# To predict the flower, whose beauty is drawn.
# The DecisionTreeClassifier is born,
# To classify Iris, from dusk until morn.

# Create a DecisionTreeClassifier instance named 'clf'.
# Decision trees are a type of supervised learning algorithm used for
# classification tasks. The classifier will learn to predict the
# species of Iris flowers based on the features provided in the
# training data.

clf = DecisionTreeClassifier()

#**************************************************#
# Train the decision tree classifier using the training data.
# The fit method fits the model to the training data, learning the
# patterns and relationships between the features and the target
# variable. The 'x_train' parameter contains the features of the
# training set, while 'y_train' contains the corresponding target
# values (species labels).

clf.fit(x_train, y_train)


# With the classifier trained, predictions we seek,
# To evaluate its performance, strong and unique.
# Using the test set, unseen before,
# We predict the labels, like a wise troubadour.

# Predict the labels for the test data using the trained classifier.
# The predict method applies the trained model to the features in the
# test set ('x_test') and returns the predicted labels ('y_predict').
# These predictions will be compared with the actual labels to assess
# the model's performance.

y_predict = clf.predict(x_test)

#**************************************************#
# Display the predicted labels.
# The predicted labels are stored in the variable 'y_predict', and
# this command prints them to the console. It allows us to examine
# the model's predictions and compare them with the actual labels
# to evaluate the classifier's accuracy.

y_predict


## Evaluation Metric

# Accuracy, a measure we seek to find,
# To judge the classifier's performance, refined.
# By comparing predicted labels with those true,
# The accuracy score reveals how well it grew.

# Compute the accuracy score of the classifier.
# The accuracy_score function from the sklearn.metrics module is used
# to compare the predicted labels ('y_predict') with the true labels
# from the test set ('y_test'). It returns a value representing the
# accuracy of the classifier's predictions.

accuracy = metrics.accuracy_score(y_test, y_predict)

# Print the accuracy score to the console.
# The accuracy score, calculated earlier, is printed to the console
# to provide insight into the classifier's performance. It indicates
# the proportion of correctly predicted labels out of all labels
# in the test set.

print("Accuracy:", accuracy)

```