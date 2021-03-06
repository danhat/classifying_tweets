
import numpy as np
import pandas as pd
import nltk
import sklearn 
import string
import re # helps  filter urls
from sklearn.metrics import accuracy_score
#from IPython.display import display, Latex, Markdown
import copy
from scipy import stats



nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')




posMapping = {
    # "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
  }


""" 
  Normalizes case and handles punctuation
  Inputs:
    text: str: raw text
    lemmatizer: an instance of a class implementing the lemmatize() method
                (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
  Outputs:
    list(str): tokenized text
"""
def process(text, lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()):
  
  
  text = text.lower()
  text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text) # remove url
  text = text.replace("'s", '') # remove possessive 's 
  text = text.replace("'", '') # remove other apostrophes
  text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) # remove punctuation
  text = re.sub(' +', ' ', text) # remove whitespace b/w words
  text = text.strip() # remove leading or trailing whitespace
  
  tokens = nltk.word_tokenize(text)
  tokens = nltk.pos_tag(tokens)

  pos_mapping = {
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
  }

  lemmatized = []
  for word, pos in tokens:
    tag = pos_mapping.get(pos[0])
    if (tag != None):
      lemmatized.append(lemmatizer.lemmatize(word, tag))
    else:
      lemmatized.append(lemmatizer.lemmatize(word))


  return lemmatized



""" 
  process all text in the dataframe using process function.
  Inputs
    df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
    lemmatizer: an instance of a class implementing the lemmatize() method
                (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
  Outputs
    pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                the output from process_text() function. Other columns are unaffected.
"""
def process_all(df, lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()):
  df['text'] = df['text'].apply(process)
  return df
    


""" 
  creates the feature matrix using the processed tweet text
  Inputs:
    tweets: pd.DataFrame: tweets read from train/test csv file, containing the column 'text'
    stop_words: list(str): stop_words by nltk stopwords
  Outputs:
    sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
        we need this to tranform test tweets in the same way as train tweets
    scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
"""
def create_features(processed_tweets, stop_words):  
  docs2 = copy.deepcopy(processed_tweets['text'])
  docs = [' '.join(i) for i in docs2]

  
  vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(use_idf = True, min_df = 2, stop_words = stop_words)
  vectors = vectorizer.fit_transform(docs)
  
  return vectorizer, vectors



""" 
  creates the class labels from screen_name
  Inputs:
    tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
  Outputs:
    numpy.ndarray(int): dense binary numpy array of class labels
  """
def create_labels(processed_tweets): 
  labels = []
  initialize = 1
  for user in processed_tweets['screen_name']:
    if (initialize == 1):
      if (user == 'realDonaldTrump' or user == 'mike_pence' or user == 'GOP'):
        labels = np.array(0, dtype = int)
      else:
        labels = np.array(1, dtype = int)
      initialize = 0
    else:
      if(user == 'realDonaldTrump' or user == 'mike_pence' or user == 'GOP'):
        labels = np.append(labels, 0)
      else:
        labels = np.append(labels, 1)

  return labels
    



"""
  A classifier that predicts the mode of training labels
"""
class MajorityLabelClassifier():

  # initializa
  def __init__(self):
    self.mode = None
  

  # Implement fit by taking training data X and their labels y and finding the mode of y  
  def fit(self, X, y):
    m = stats.mode(y)
    self.mode = int(m[0])
    
  
  # Implement to give the mode of training labels as a prediction for each data instance in X return labels
  def predict(self, X):
    labels = []
    for i in X:
      labels.append(self.mode)
    
    return labels
    





"""
  learns a classifier from the input features and labels using the kernel function supplied
  Inputs:
    X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features()
    y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_labels()
    kernel: str: kernel function to be used with classifier. [linear|poly|rbf|sigmoid]
  Outputs:
    sklearn.svm.classes.SVC: classifier learnt from data
""" 
def learn_classifier(X_train, y_train, kernel):  
  classifier = sklearn.svm.SVC(kernel = kernel)
  classifier.fit(X_train, y_train)
  return classifier
  




""" 
  evaluates a classifier based on a supplied validation data
  Inputs:
    classifier: sklearn.svm.classes.SVC: classifer to evaluate
    X_validation: scipy.sparse.csr.csr_matrix: sparse matrix of features
    y_validation: numpy.ndarray(int): dense binary vector of class labels
  Outputs:
    double: accuracy of classifier on the validation data
"""
def evaluate_classifier(classifier, X_validation, y_validation):
  predictions = classifier.predict(X_validation)
  accuracy = accuracy_score(y_validation, predictions) 
  return accuracy
 

"""
  Select the kernel giving best results using k-fold cross-validation.
  Other parameters should be left default.
  Input:
    kf (sklearn.model_selection.KFold): kf object defined above
    X (scipy.sparse.csr.csr_matrix): training data
    y (array(int)): training labels
  Return:
    best_kernel (string)
"""
def best_model_selection(kf, X, y):
  best_kernel = 'linear'
  best_accuracy = 0
  
  for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    accuracy = []
    for train, test in kf.split(X):
      # Split train-test
      X_train, X_test = X[train], X[test]
      y_train, y_test = y[train], y[test]
      classifier = learn_classifier(X_train, y_train, kernel)
      accuracy.append(evaluate_classifier(classifier, X_test, y_test))
    
    if (sum(accuracy) / 4 > best_accuracy):
      best_accuracy = sum(accuracy) / 4
      best_kernel = kernel
    
  return best_kernel





""" 
  predicts class labels for raw tweet text
  Inputs:
    tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
    classifier: sklearn.svm.classes.SVC: classifier learnt
    unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
  Outputs:
    numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
"""
def classify_tweets(tfidf, classifier, unlabeled_tweets): 
  tweets = pd.read_csv("tweets_train.csv", na_filter=False)
  train_tweets = process_all(tweets)
  tfidf, X = create_features(train_tweets, stopwords)
  y = create_labels(train_tweets)
  
  classifier = learn_classifier(X, y, 'poly')
  
  test_tweets = process_all(unlabeled_tweets)
  tfidf, X_test = create_features(test_tweets, stopwords)
  classifications = classifier.predict(X_test)
  
  return classifications




def main():
  tweets = pd.read_csv("tweets_train.csv", na_filter=False)
  print('/nhead of tweets: ')
  print(tweets.head())
  
  processed_tweets = process_all(tweets)
  print('/nhead of processed tweets: ')
  print(processed_tweets.head())
  
  (tfidf, X) = create_features(processed_tweets, stopwords)
  print('/n shape of : ', X.shape)
  
  y = create_labels(processed_tweets)
  print('/nsize of processed tweets labels: ', len(y))
  
  classifier = learn_classifier(X, y, 'linear')  
  accuracy = evaluate_classifier(classifier, X, y)
  print('/naccuracy of classifier: ', accuracy) 
  # should give 0.956700196554515
  
  kf = sklearn.model_selection.KFold(n_splits=4, random_state=1, shuffle=True)
  kf
  
  best_kernel = best_model_selection(kf, X, y)
  print(best_kernel)




if __name__ == '__main__':
  main()






