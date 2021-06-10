# Airline-passenger-referral-prediction
## OBJECTIVE :

The main objective of this project is to predict whether passengers will refer the airline to their friends. In this project we deployed multiple machine learning models to see the performance.  

## UNDERSTANDING DATA:



### Description of features 

* airline: Name of the airline.
* overall: Overall point is given to the trip between 1 to 10.
* author: Author of the trip
* review date: Date of the Review customer review: Review of the customers in free text format
* aircraft: Type of the aircraft
* traveller type: Type of traveler (e.g. business, leisure)
* cabin: Cabin at the flight date flown: Flight date
* seat comfort: Rated between 1-5
* cabin service: Rated between 1-5
* foodbev: Rated between 1-5 entertainment: Rated between 1-5
* groundservice: Rated between 1-5
* value for money: Rated between 1-5
* recommended: Binary, target variable.


### The shape of data is (131895,17)


# Imputation of Target Variable

By using NLP On text column and then using **Naive Bayes Classifier** we were able to Impute missing values in target variable.   

 





##  VADER (Valence Aware Dictionary for Sentiment Reasoning) :

 Introduction and Using VADER :

To do sentimental analysis  we will use NLTK (Natural Language Toolkit), more specifically we use a tool named VADER.
In essence, it analyses a text and returns a dictionary with four keys. The letters 'neg', 'neu', and 'pos' stand for 'Negative,"Neutral,' and Positive,' respectively.
The final key is called compound, and it is a mix of the previous three here it is termed as polarity.
To see the source code of VADER https://www.nltk.org/_modules/nltk/sentiment/vader.html here.
Using VADER to do sentimental analysis on review text and to extract sentiment of positive and negative out of it.


### Adding a new feature rec_nonrec:

rec_nonrec - A polarity scoring rate feature. If the score is less than 0.7, the sentiment is negative and is filled with zero; if the score is greater than 0.7, the sentiment is positive and is filled with one.
 classifying the polarity into one of the 2 categories: Positive, Negative 









#  WORKING WITH DIFFERENT MODELS

## Models used are 
## Logistic Regression
## DecisionTree
## Random Forest
## Gradient Boosting
## XG Boost
## SVM




## WORKING WITH ANOMALIES :
We can find some suspicious reviews in the data set where the overall score is 9 or 10 but the recommended column is blank, or where the overall score is 1 or 2 but the recommended column is blank. These are anomalies in our data set.
We swapped them out for the right ones. On both the test and train sets, the results have improved by nearly 2%.






# CONCLUSIONS
We have built classifier models using 7 different types of classifiers and all these are able to give accuracy of more than 95%.
The most important features are Overall rating and Value for money that contribute to a model's prediction.
The classifier model developed will enable airlines ability to identify impactful passengers who can help in bringing more revenue.

