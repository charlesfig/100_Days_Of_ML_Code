from sklearn import datasets
from sklearn.svm import SVC
#SV Classifier for classifying the digits
from scipy import misc
#for classifying the 


digits = datasets.load_digits()
features = digits.data
labels = digits.target



clf = SVC(gamma=0.001)
#We feed data to classifier
clf.fit(features,labels)
#print(features, labels)
#Above shows features of each number 0-9 and labels, i.e. numbers 0-9
#The last one is the data for 8

#We can now predict output. We are using features of the last rule
#It should predict 8

#print(clf.predict([features[-1]]))

#We could test for 9 with the following:
#print(clf.predict([features[-2]]))

#The classifier has now been trained

#We look at the format of the data
#print(features.shape)
#We see there are 64 columns, i.e. 64 pixels, meaning 8x8 image

#We load the image
img = misc.imread("8.jpg")
#We resize the image
img = misc.imresize(img, (8,8))
#print(img)
#The above shows 8 rows with and 8 pixels each. We see the array is multidimensional

#print(img.dtype) 
#Above shows dataype of our image as uint8
#print(features.dtype)
#Above shows datatype of digits.data (i.e. our features) as float64
#We have to convert our image's datatype
img = img.astype(digits.images.dtype)

#One last thing
#print(features[-1])
#The above shows the trained data has 0-16 values ?
#print(img)
#Our image is 0-255 because is a 256 bit image. We need to convert it/scale
img = misc.bytescale(img, high=16, low=0)

#Now we need to convert this 8x8 matrix into 1 dimension, i.e 1x64 matrix
#Need to use 2 loops
x_test = []

for eachRow in img:
	for eachPixel in eachRow:
		x_test.append(sum(eachPixel/3.0))
		#Because there are 3 informations for each pixel

#print(x_test)
#The above shows the 1x64 matrix in which the image has been converted

#The image is now ready to be analyse by the classifier
print(clf.predict([x_test]))

#Model is far from perfect. Can be due to the trained data and the input
#As we saw in Line 32, there are only 1797 datapoints in our training set
