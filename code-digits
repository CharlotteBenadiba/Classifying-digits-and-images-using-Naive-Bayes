from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
labels = []
images = []
i=0
for l in Y_train:
  if l not in labels:
    labels.append(l)
    images.append(X_train[i])
  i+=1
num=10
num_row = 2
num_col = 5
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()

X = np.concatenate([X_train, X_test])
X = X.reshape((70000,784))
y = np.concatenate([Y_train, Y_test])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1/7)

from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import accuracy_score

x_train = x_train.reshape((60000,784))
x_test = x_test.reshape((10000,784))
BernNB = BernoulliNB(binarize= True)
BernNB.fit(x_train, y_train)
y_true = y_test
y_pred = BernNB.predict(x_test)
print("Accuracy Score: ",accuracy_score(y_true,y_pred))

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

params = BernNB.coef_
num=10
num_row = 2
num_col = 5
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    avgImg = params[i]
    ax = axes[i//num_col, i%num_col]
    ax.imshow(avgImg.reshape((28,28)), cmap='gray')
    ax.set_title('Label: {}'.format(i))
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt



confusion_matrix = confusion_matrix(y_test, y_pred)
diagonal_sum = np.sum(np.diag(confusion_matrix))

# Plot the confusion matrix
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
plot_confusion_matrix(BernNB, x_test, y_test)

# Finally, divide the diagonal sum by the total number of elements in the matrix
total_accuracy = diagonal_sum / confusion_matrix.size
print("Accuracy: ",total_accuracy,"\n\n")

