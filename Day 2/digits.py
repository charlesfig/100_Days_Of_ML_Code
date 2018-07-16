from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()

#digits.data gives access to the features that can be used to classify the digits samples
#digits.target gives truth for digitdataset, the number corresponding to each digit image that we are trying to learn

#We use the estimator sklearn.svm.SVC
#An estimator for classification is a Python object that implements the methods fit(X,y) and predict(T)
from sklearn import svm

clf=svm.SVC(gamma=0.001, C=100.)
#Here we set the value of gamma manually. 
#We could find good values for the parameters by using tools such as grid search and cross validation
#We are calling our estimator clf (for classfier)
#We need to fit it to the model, i.e. it must learn from the model
#We do this by passing training set to the fit method
#For our training set, we use all the images of our dataset but the last one, the last entry of digits.data
features = digits.data[:-1]
labels = digits.target[:-1]
clf.fit(features, labels)
#We can now predict new values. 
#We ask the classfier what is the digit of our last image in the digits dataset (not used to train it)
prediction = clf.predict(digits.data[-1:])
#We print the answer. Given as (8)
print(prediction)