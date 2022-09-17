#!/usr/bin/env python3
"""
File name: dataPreparation.py
"""

import pandas as pd # import library to read data into dataframe
pd.set_option("display.max_columns", None)
import numpy as np # import numpy library
import re # import library for regular expression
import random # library for random number generation
import csv

#%matplotlib inline
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

# If running locally, you can try using the graphviz library but we'll use sklearn's plot_tree method
# !conda install python-graphviz --yes
# from sklearn.tree import export_graphviz

import itertools

# Uncomment if running locally, else download data using the following code cell
# recipes = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0103EN-SkillsNetwork/labs/Module%202/recipes.csv")
# print("Data read into dataframe!") # takes about 30 seconds
# recipes.to_csv('raw_data.csv', index = False)

#uncomment once you have downloaded and saved the data in the previous cell block 
recipes = pd.read_csv("raw_data.csv")
################################### Data Modeling #####################################
freqTable = recipes["country"].value_counts()	#frecuency table
print(freqTable)

#fixing the name of the columns showing the cuisine
column_names = recipes.columns.values
column_names[0] = "cuisine"
recipes.columns = column_names

print(recipes)

#Making all the cuisine names lower case
recipes["cuisine"] = recipes["cuisine"].str.lower()


#Making the cuisine names consistent	
recipes.loc[recipes["cuisine"] == "austria", "cuisine"] = "austrian"
recipes.loc[recipes["cuisine"] == "belgium", "cuisine"] = "belgian"
recipes.loc[recipes["cuisine"] == "china", "cuisine"] = "chinese"
recipes.loc[recipes["cuisine"] == "canada", "cuisine"] = "canadian"
recipes.loc[recipes["cuisine"] == "netherlands", "cuisine"] = "dutch"
recipes.loc[recipes["cuisine"] == "france", "cuisine"] = "french"
recipes.loc[recipes["cuisine"] == "germany", "cuisine"] = "german"
recipes.loc[recipes["cuisine"] == "india", "cuisine"] = "indian"
recipes.loc[recipes["cuisine"] == "indonesia", "cuisine"] = "indonesian"
recipes.loc[recipes["cuisine"] == "iran", "cuisine"] = "iranian"
recipes.loc[recipes["cuisine"] == "italy", "cuisine"] = "italian"
recipes.loc[recipes["cuisine"] == "japan", "cuisine"] = "japanese"
recipes.loc[recipes["cuisine"] == "israel", "cuisine"] = "israeli"
recipes.loc[recipes["cuisine"] == "korea", "cuisine"] = "korean"
recipes.loc[recipes["cuisine"] == "lebanon", "cuisine"] = "lebanese"
recipes.loc[recipes["cuisine"] == "malaysia", "cuisine"] = "malaysian"
recipes.loc[recipes["cuisine"] == "mexico", "cuisine"] = "mexican"
recipes.loc[recipes["cuisine"] == "pakistan", "cuisine"] = "pakistani"
recipes.loc[recipes["cuisine"] == "philippines", "cuisine"] = "philippine"
recipes.loc[recipes["cuisine"] == "scandinavia", "cuisine"] = "scandinavian"
recipes.loc[recipes["cuisine"] == "spain", "cuisine"] = "spanish_portuguese"
recipes.loc[recipes["cuisine"] == "portugal", "cuisine"] = "spanish_portuguese"
recipes.loc[recipes["cuisine"] == "switzerland", "cuisine"] = "swiss"
recipes.loc[recipes["cuisine"] == "thailand", "cuisine"] = "thai"
recipes.loc[recipes["cuisine"] == "turkey", "cuisine"] = "turkish"
recipes.loc[recipes["cuisine"] == "vietnam", "cuisine"] = "vietnamese"
recipes.loc[recipes["cuisine"] == "uk-and-ireland", "cuisine"] = "uk-and-irish"
recipes.loc[recipes["cuisine"] == "irish", "cuisine"] = "uk-and-irish"

print(recipes)

#now we remove cuisines lists with less than 50 recipes
recipes_counts = recipes["cuisine"].value_counts()
cuisines_index = recipes_counts >50
cuisines_keep = list(np.array(recipes_counts.index.values)[np.array(cuisines_index)])

rows_before = recipes.shape[0]	#number of rows of original dataframe
print("Number of rows of original dataframe")

recipes = recipes.loc[recipes["cuisine"].isin(cuisines_keep)]

rows_after = recipes.shape[0]	#number of rows of processed dataframe
print("Number of rows of processed dataframe is {}".format(rows_after))
print("{} rows removed!".format(rows_before - rows_after))

#Convert all the yes's to 1 and the no's to 0
recipes = recipes.replace(to_replace = "Yes", value = 1)
recipes = recipes.replace(to_replace = "No", value = 0)

################################### Data Modeling #####################################

#Building a decision tree based on Asian and Indian Cuisine
#select subset of cuisines
asian_indian_recipes = recipes[recipes.cuisine.isin(["korean", "japanes", "chinese", "thai", "indian"])]
cuisines = asian_indian_recipes["cuisine"]
ingredients = asian_indian_recipes.iloc[:,1:]

bamboo_tree = tree.DecisionTreeClassifier(max_depth = 3)
bamboo_tree.fit(ingredients, cuisines)
print("Decision tree model saved to bamboo_tree!")
print("##############################################################")

#plotting the decision tree

# if you're using the graphviz library, you can run these lines of code. Otherwise, this is configured to use plot_tree from sklearn
# export_graphviz(bamboo_tree,
#                 feature_names=list(ingredients.columns.values),
#                 out_file="bamboo_tree.dot",
#                 class_names=np.unique(cuisines),
#                 filled=True,
#                 node_ids=True,
#                 special_characters=True,
#                 impurity=False,
#                 label="all",
#                 leaves_parallel=False)

# with open("bamboo_tree.dot") as bamboo_tree_image:
#     bamboo_tree_graph = bamboo_tree_image.read()
# graphviz.Source(bamboo_tree_graph)

recipes.head()	#show the first few rows
print(recipes.head())
plt.figure(figsize=(20,10))  # customize according to the size of your tree
_ = tree.plot_tree(bamboo_tree,
                   feature_names = list(ingredients.columns.values),
                   class_names=np.unique(cuisines),filled=True,
                   node_ids=True,
                   impurity=False,
                   label="all",
                   fontsize=10, rounded = True)
plt.show()
recipes.shape	#get the dimensions of the data frame
print("dimension of data set",recipes.shape)

#So our data set consists of 57,691 recipes
#Checking if the following ingredients exists in the data set

################################### Data Evaluation #####################################

#create a new data frame only using the data pertaining to the asian and the indian cuisines
bamboo = recipes[recipes.cuisine.isin(["korean", "japanese", "chinese", "thai", "indian"])]

#seeing how many recipes are for each cuisine type
bamboo_count = bamboo["cuisine"].value_counts()
print(bamboo_count)

#removing 30 recipes from each cuisime to use as a test
sample_n = 30
random.seed(1234)
bamboo_test = bamboo.groupby("cuisine", group_keys = False).apply(lambda x: x.sample(sample_n))
bamboo_test_ingredients = bamboo_test.iloc[:,1:]	#ingredients
bamboo_test_cuisines = bamboo_test["cuisine"]	#corresponding labels

#checking that there are 30 recipes for each cuisine
print("recipes in the cuisine test data frame", '\n' ,bamboo_test["cuisine"].value_counts())

#Creating the training set
bamboo_test_index = bamboo.index.isin(bamboo_test.index)
bamboo_train = bamboo[~bamboo_test_index]
bamboo_train_ingredients  =bamboo_train.iloc[:,1:]
bamboo_train_cuisines = bamboo_train["cuisine"]

bamboo_train["cuisine"].value_counts()

#building decision tree using the training set | bamboo_train_tree for prediction
bamboo_train_tree = tree.DecisionTreeClassifier(max_depth=15)
bamboo_train_tree.fit(bamboo_train_ingredients, bamboo_train_cuisines)

print("Decision tree model saved to bamboo_train_tree!")

#plotting decision tree model 
plt.figure(figsize=(20,10))  # customize according to the size of your tree
_ = tree.plot_tree(bamboo_train_tree,
                   feature_names=list(bamboo_train_ingredients.columns.values),
                   class_names=np.unique(bamboo_train_cuisines),
                   filled=True,
                   node_ids=True,
                   impurity=False,
                   label="all",
                   fontsize=5, rounded = True)
plt.show()

#testing the data model on the test data
bamboo_pred_cuisines = bamboo_train_tree.predict(bamboo_test_ingredients)

#To quantify how well the decision tree is able to determine the cuisine of each recipe correctly, 
#we will create a confusion matrix which presents a nice summary on how many recipes from each 
#cuisine are correctly classified. It also sheds some light on what cuisines are being confused 
#with what other cuisines.

test_cuisines = np.unique(bamboo_test_cuisines)
bamboo_confusion_matrix = confusion_matrix(bamboo_test_cuisines, bamboo_pred_cuisines, labels = test_cuisines)
title = 'Bamboo Confusion Matrix'
cmap = plt.cm.Blues

plt.figure(figsize=(8, 6))
bamboo_confusion_matrix = (
    bamboo_confusion_matrix.astype('float') / bamboo_confusion_matrix.sum(axis=1)[:, np.newaxis]
    ) * 100

plt.imshow(bamboo_confusion_matrix, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(test_cuisines))
plt.xticks(tick_marks, test_cuisines)
plt.yticks(tick_marks, test_cuisines)

fmt = '.2f'
thresh = bamboo_confusion_matrix.max() / 2.
for i, j in itertools.product(range(bamboo_confusion_matrix.shape[0]), range(bamboo_confusion_matrix.shape[1])):
    plt.text(j, i, format(bamboo_confusion_matrix[i, j], fmt),
             horizontalalignment="center",
             color="white" if bamboo_confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()






