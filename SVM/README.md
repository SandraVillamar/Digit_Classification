# Digit Classification via SVM
In this problem, we will use the SVM to, once again, classify digits. In all questions, you will use the training set contained in the directory MNISTtrain (60,000 examples) and the test set in the directory MNISTtest (10,000). To reduce training time, we will only use 20,000 training examples. To read the data, you should use the script readMNIST.m (use readDigits=20,000 or readDigits=10,000 respectively, and offset=0). This returns two matrices. The first (imgs) is a matrix with n ∈ {10000, 20000} rows, where each row is a 28 × 28 image of a digit, vectorized into a 784-dimensional row vector. The second (labels) is a matrix with n ∈ {10000, 20000} rows, where each row contains the class label for the corresponding image in imgs. Since there are 10 digit classes, we will learn 10 binary classifiers. Each classifier classifies one class against all others. For example, classifier 1 assigns label Y = 1 to the images of class 1 and label Y = −1 to images of all other classes. Download
and install the libsvm package to learn the SVM classifiers.

a) In this problem, we will learn linear SVMs. Using libsvm, learn three SVMs with values of the regularization constant C ∈ {2, 4, 8}. For each classifier and digit, 1) report the test error, 2) report the number of support vectors, and 3) plot the three support vectors of largest Lagrange multiplier on each side of the boundary. For each classifier, report the overall classification error. Comment on the results.

b) For each binary classifier, make a plot of cumulative distribution function (cdf) of the margins y<sub>i</sub>(w<sup>T</sup> x<sub>i</sub> + b) of all training examples. Comment on the results.

c) Repeat a) and b) for an SVM with radial basis function kernel. Use the script grid.py, included in the libsvm package, to cross-validate the values of C and γ, and then use the two parameters for the classification. Compare the results with those of a) and b).

# Results
![image](https://user-images.githubusercontent.com/15370068/161126684-4c55dced-d195-44fb-a23d-392e116233ea.png)
![image](https://user-images.githubusercontent.com/15370068/161126712-4bb3b18b-a0e7-4731-9382-a812c7182a07.png)
![image](https://user-images.githubusercontent.com/15370068/161126737-c884a221-5e91-49e0-99f5-04bdfe9720d3.png)
![image](https://user-images.githubusercontent.com/15370068/161126759-a3d940bf-0312-47b5-b930-6391295825f9.png)
![image](https://user-images.githubusercontent.com/15370068/161126834-9187d4c2-dadc-42ec-a535-df5e09252043.png)
![image](https://user-images.githubusercontent.com/15370068/161126873-b7db1d0d-1108-48ba-b01a-e9a937725433.png)
![image](https://user-images.githubusercontent.com/15370068/161126901-0c230c06-5b62-49a3-8c51-66b472144f77.png)
