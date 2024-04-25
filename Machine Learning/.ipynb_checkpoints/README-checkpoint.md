
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

```

### Applied machine learning
```python

```