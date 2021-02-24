# Classifying Tweets (Supervised Machine Learning)


## Table of contents
* [General Info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Sources](#sources)


## General Info
In this problem, Twitter data is analyzed and extracted using the [twitter API](https://dev.twitter.com/overview/api). The data contains tweets posted by the following six Twitter accounts: `realDonaldTrump, mike_pence, GOP, HillaryClinton, timkaine, TheDemocrats`

For every tweet, there are two pieces of information:
- `screen_name`: the Twitter handle of the user tweeting and
- `text`: the content of the tweet.

The tweets have been divided into two parts - train and test available in CSV files. For train, both the `screen_name` and `text` attributes are provided but for test, `screen_name` is hidden.

The goal of the problem is to "predict" the political inclination (Republican/Democratic) of the Twitter user from one of his/her tweets. The ground truth (true class labels) is determined from the `screen_name` of the tweet as follows
- `realDonaldTrump, mike_pence, GOP` are Republicans
- `HillaryClinton, timkaine, TheDemocrats` are Democrats

This is a binary classification problem. 

The problem proceeds in three stages:
- **Text processing**: clean up the raw tweet text using the various functions offered by the [nltk](http://www.nltk.org/genindex.html) package.
- **Feature construction**: construct bag-of-words feature vectors and training labels from the processed text of tweets and the `screen_name` columns respectively.
- **Classification**: use [sklearn](http://scikit-learn.org/stable/modules/classes.html) package to learn a model which classifies the tweets as desired. 


## Technologies
* Python 3.7
* Anaconda3
* pip


## Setup
```
$ pip install nltk
$ pip install sklearn
$ pip install <other libraries not downloaded>
$ cd ../tweets
$ py tweets.py
```

or open the jupyter notebook [here](tweets.ipynb)



## Sources
* [twitter API](https://dev.twitter.com/overview/api)
* [nltk](http://www.nltk.org/genindex.html) 
* [sklearn](http://scikit-learn.org/stable/modules/classes.html)
