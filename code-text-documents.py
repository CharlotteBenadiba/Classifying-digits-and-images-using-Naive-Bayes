import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from math import log
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_data = fetch_20newsgroups(subset='train', categories=None, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))


# Split the data and target into train and test sets
X_train, X_test, y_train, y_test = train_test_split(train_data.data, train_data.target, test_size=0.2)

# Create the CountVectorizer
vectorizer = CountVectorizer(stop_words='english')

# Transform the text data into numerical vectors
vectors = vectorizer.fit_transform(X_train)

# Create the MultinomialNB classifier
classifier = MultinomialNB()

# Train the classifier on the training data
classifier.fit(vectors, y_train)

# Make predictions on new data
new_vectors = vectorizer.transform(X_test)
predictions = classifier.predict(new_vectors)
print("Accuracy Score of MultinomialNB with CountVectorizer: ",accuracy_score(y_test,predictions))

# Create the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Transform the text data into numerical vectors
vectors = vectorizer.fit_transform(X_train)

# Train the classifier on the training data
classifier.fit(vectors, y_train)

# Make predictions on new data
new_vectors = vectorizer.transform(X_test)
predictions = classifier.predict(new_vectors)
print("Accuracy Score of MultinomialNB with TfidfVectorizer: ",accuracy_score(y_test,predictions))

# Load the 20 Newsgroups dataset
train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), random_state=42)

# data = fetch_20newsgroups()
X, y = train_data.data, train_data.target

# Create the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Transform the text data into numerical vectors
vectors = vectorizer.fit_transform(X)

# Create the multinomial Naive Bayes model
model = MultinomialNB()

# Use the learning_curve() function to plot a learning curve
train_sizes, train_scores, val_scores = learning_curve(
    model, vectors, y, cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.2, 1.0, 5)
)

# Plot the learning curve
plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score")
plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation score")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy Score")
plt.legend()
plt.show()

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class NaiveBayes(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        """Fit the model to the data X and labels y.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training data.
        
        y : array-like, shape (n_samples,)
            The target labels.
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Calculate the class priors
        self.class_counts = np.bincount(y)
        self.priors = self.class_counts / len(y)
        
        # Calculate the feature counts for each class
        self.feature_counts = []
        for c in range(len(self.class_counts)):
            mask = (y == c)
            self.feature_counts.append(X[mask,:].sum(axis=0))
        
        # Calculate the total feature counts for each class
        self.total_feature_counts = np.sum(self.feature_counts, axis=1)        
        return self
    
    def predict(self, X):
        """Predict the class for each sample in X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        
        Returns
        -------
        y_pred : array, shape (n_samples,)
            The predicted class for each sample.
        """
        y_pred = []
        for xi in X:
            # Calculate the log probability of each class
            
            log_probs = np.log(self.priors)
            for c in range(len(self.class_counts)):
                log_probs[c] += np.sum((xi * np.log(self.feature_counts[c] + 1).transpose()) - np.log(self.total_feature_counts[c] + X.shape[1]))
            
            # Predict the class with the highest log probability
            y_pred.append(np.argmax(log_probs))
        
        return y_pred

newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
cv = CountVectorizer(stop_words="english", max_features=1000) ## we limited max features for performance
nb = NaiveBayes()
x = cv.fit_transform(newsgroups_train)
pipeline = make_pipeline(cv,nb)
#print(newsgroups_train.target)
pipeline.fit(newsgroups_train.data, newsgroups_train.target)
y_pred = pipeline.predict(newsgroups_test.data)
print("Accuracy Score of NaiveBayes with CountVectorizer: ",accuracy_score(y_pred,newsgroups_test.target))

ewsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
tf = TfidfVectorizer(stop_words="english", max_features=1000) ## we limited max features for performance
nb = NaiveBayes()
x = tf.fit_transform(newsgroups_train)
pipeline = make_pipeline(tf,nb)
#print(newsgroups_train.target)
pipeline.fit(newsgroups_train.data, newsgroups_train.target)
y_pred = pipeline.predict(newsgroups_test.data)
print("Accuracy Score of NaiveBayes with CountVectorizer: ",accuracy_score(y_pred,newsgroups_test.target))

import string
def print_txt(txt, hot, cold):
  """
  print the text, coloring hot and cold words with colors
  """
  cold_color='\x1b[41;37m{}\x1b[0m'#red color
  hot_color='\x1b[42;37m{}\x1b[0m'#green color
  def color(token):
    lower = str(token).lower()
    lower = lower.replace('\t','').replace('\n','')
    lower = lower.translate(string.punctuation)
    if (lower in hot) and (lower in cold):
      return token
    elif lower in hot:
      return hot_color.format(token)
    elif lower in cold:
      return cold_color.format(token)
    else:
      return token
  colored_txt = " ".join([color(token) for token in txt.split(' ')])
  print(colored_txt)
print_txt('This word support the first class but this the other', ['word'],['other'])

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def calc_p_c_given_xi(model):
  # Load the 20 Newsgroups dataset
  data = fetch_20newsgroups()
  X = data['data']
  y = data['target']

  # Extract the features using a CountVectorizer
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(X)

  # Train a MultinomialNB model
  
  model.fit(X, y)

  # Get the feature names
  feature_names = vectorizer.get_feature_names()

  # Select a class and get the log probabilities of the features given that class
  class_idx = 0
  coef = model.coef_[class_idx]

  # Sort the log probabilities in descending order
  sorted_coef_idx = coef.argsort()[::-1]

  # Get the top 10 words that support the correct class
  top_20_correct = [feature_names[i] for i in sorted_coef_idx[:20]]

  # Get the top 10 words that support the wrong class
  top_20_wrong = [feature_names[i] for i in sorted_coef_idx[-20:]]

  for i in range(0,20):
    
    txt = "word: {} support | word: {} does'nt support class".format(top_20_correct[i],top_20_wrong[i])
    print_txt(txt, top_20_correct[i],top_20_wrong[i])
  
calc_p_c_given_xi(MultinomialNB())

