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

# import piplite
# await piplite.install(['skilsnetwork'])
# import skillsnetwork


# Uncomment if running locally, else download data using the following code cell
 recipes = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0103EN-SkillsNetwork/labs/Module%202/recipes.csv")
 print("Data read into dataframe!") # takes about 30 seconds
 recipes.to_csv('raw_data.csv', index = False)

#uncomment once you have downloaded and saved the data in the previous cell block 
recipes = pd.read_csv("raw_data.csv")

# await skilsnetwork.download_dataset(
# 	 "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0103EN-SkillsNetwork/labs/Module%202/recipes.csv",
#     "recipes.csv")

# recipes = pd.read_csv("recipes.csv")
# print("Data read into dataframe!")

recipes.head()	#show the first few rows
print(recipes.head())

recipes.shape	#get the dimensions of the data frame
print("dimension of data set",recipes.shape)

#So our data set consists of 57,691 recipes
#Checking if the following ingredients exists in the data set

ingredients = list(recipes.columns.values)

print([match.group(0) for ingredient in ingredients for match in [(re.compile(".*(rice).*")).search(ingredient)] if match])
print([match.group(0) for ingredient in ingredients for match in [(re.compile(".*(wasabi).*")).search(ingredient)] if match])
print([match.group(0) for ingredient in ingredients for match in [(re.compile(".*(soy).*")).search(ingredient)] if match])

#----------------- Data Preparation-------------------

#Check if the data set needs cleaning
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


#We are going to run the recipes that have rice&&soy and wasabi && seaweed
recipes.head()
check_recipes = recipes.loc[
(recipes["rice"] == 1) & (recipes["soy_sauce"] == 1) & (recipes["wasabi"] == 1) & (recipes["seaweed"] == 1)]

print(check_recipes)

#Counting the ingredients in all the recipes
ingredients_col = recipes.iloc[:, 1:].sum(axis = 0) 

#define each column as a panda series
ingredient = pd.Series(ingredients_col.index.values, index = np.arange(len(ingredients_col)))
count = pd.Series(list(ingredients_col), index = np.arange(len(ingredients_col)))

#create the dataframe
ingredient_df = pd.DataFrame(dict(ingredient = ingredient, count = count))
ingredient_df = ingredient_df[["ingredient", "count"]]
print(ingredient_df.to_string())

#Sorting the dataframe of ingredients and their total counts
ingredient_df.sort_values(["count"], ascending = False, inplace = True)
ingredient_df.reset_index(inplace = True, drop = True)
print(ingredient_df)

#Creating a profile for each Cuisine (seeing what each type of cuisine uses wich ingredient the most)
cuisines = recipes.groupby("cuisine").mean()
cuisines.head()

num_ingredients = 4

print("####################### Top Ingredients #######################")
def print_top_ingredients(row):
	print("##############################################################")
	print(row.name.upper())
	row_sorted = row.sort_values(ascending = False)*100
	top_ingredients = list(row_sorted.index.values)[0:num_ingredients]

	for ingredients_col, ingredient in enumerate(top_ingredients):
		print("%s (%d%%)" % (ingredient, row_sorted[ingredients_col]), end = ' ')
	print("\n")
	print("##############################################################")

create_cuisines_profiles = cuisines.apply(print_top_ingredients, axis =1)
