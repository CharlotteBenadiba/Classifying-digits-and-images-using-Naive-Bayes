# Classifying-digits-and-images-using-Naive-Bayes

## Table of Contents

- [1. Classifying Digits](#1.-Classifying-Digits)
- [2. Classifing Text Documents using Multinomial Naive Bayes](#2.-Classifing-Text-Documents-using-Multinomial-Naive-Bayes)

----

## 1. Classifying Digits

In this part we will test digits classification on the MNIST dataset, using Bernoulli Naive Bayes (a generative model). The MNIST dataset contains 28x28 grayscale images of handwritten digits between 0 and 9 (10 classes). For mathmatical analysis clarity, and for matching expected API, each image faltten to create a 1D array with 784 elements.
We use cross-validation to evaluate the performance of the classifier at different threshold values :

![Screenshot](/images/cross-validation-digits.png)

## 2. Classifing Text Documents using Multinomial Naive Bayes

In this section, we will classify the "20 newsgroups" data set using your own naive bayes classifier and compare to the scikit learn built in version.
The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics split in two subsets: one for training (or development) and the other one for testing (or for performance evaluation). The split between the train and test set is based upon messages posted before and after a specific date.

![Screenshot](/images/learning-curve.png)

Here we can see which top words support the correct class and which support the wrong class : 

![Screenshot](/images/top-words.png)








