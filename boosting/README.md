# Digit Classification via Boosting
In this problem, we will use boosting to classify digits. In all questions, you will use the training set contained in the directory MNISTtrain (60,000 examples) and the test set in the directory MNISTtest (10,000). To reduce training time, we will only use 20,000 training examples. To read the data, you should use the script readMNIST.m (use readDigits=20,000 or readDigits=10,000 respectively, and offset=0). This returns two matrices. The first (imgs) is a matrix with n ∈ {10000, 20000} rows, where each row is a 28 × 28 image of a digit, vectorized into a 784-dimensional row vector. The second (labels) is a matrix with n ∈ {10000, 20000} rows, where each row contains the class label for the corresponding image in imgs. Since there are 10 digit classes, we will learn 10 binary classifiers. Each classifier classifies one class against all others. For example, classifier 1 assigns label Y = 1 to the images of class 1 and label Y = −1 to images of all other classes. Boosting is then used to learn an ensemble rule:

![image](https://user-images.githubusercontent.com/15370068/161121129-21fc8b4f-3e0a-4baf-b197-4176e1af71e0.png)

After we learn the 10 rules g<sub>i</sub>(x), i ∈ {1, . . . , 10}, we assign the image to the class of largest score:
![image](https://user-images.githubusercontent.com/15370068/161121397-10d620a9-561b-4b5d-a70e-44d4efdfe9ba.png)

This implementation is called the “one-vs-all” architecture. We will use decision stumps
![image](https://user-images.githubusercontent.com/15370068/161121479-cd20283c-945f-4993-9b1e-4227e319755f.png)

as weak learners. For each weak learner, also consider the “twin” weak learner of opposite polarity:
![image](https://user-images.githubusercontent.com/15370068/161121708-2c170cfa-d157-4295-8c2c-80ea4851f7ef.png)

This simply chooses the “opposite label” from that selected by u(x; j, t). Note that j ∈ {1, . . . , 784}. For thresholds, use
![image](https://user-images.githubusercontent.com/15370068/161121814-578f7b0b-e768-4356-b538-4e2a48460eda.png)

a) Run AdaBoost for K = 250 iterations. For each binary classifier, plot train and test errors vs. iteration. Report the test error of the final classifier. For each iteration k, store the index of the example of largest weight, i.e. i<sup>*</sup> = arg max<sub>i</sub> w<sub>i</sub><sup>(k)</sup>. At iterations {5, 10, 50, 100, 250} store the margin γ<sub>i</sub> = γ(x<sub>i</sub>) of each example.

b) For each binary classifier, make a plot of cumulative distribution function (cdf) of the margins γ<sub>i</sub> of all training examples after {5, 10, 50, 100, 250} iterations (the cdf is the function F(a) = P(γ ≤ a)). Comment on what these plots tell you about what boosting is doing to the margins.

c) We now visualize the weighting mechanism. For each of the 10 binary classifiers, do the following:
- Make a plot of the index of the example of largest weight for each boosting iteration.
- Plot the three “heaviest” examples. These are the 3 examples that were most frequently selected, across boosting iterations, as the example of largest weight.
Comment on what these examples tell you about the weighting mechanism.

d) We now visualize what the weak learners do. Consider the weak learners α<sub>k</sub>, k ∈ {1, . . . , K}, chosen by each iteration of AdaBoost. Let i(k) be the index of the feature x<sub>i(k)</sub> selected at time k. Note that i(k) corresponds to an image (pixel) location. Create a 28 × 28 array a filled with the value 128. Then, for α<sub>k</sub>, k ∈ {1, . . . , K}, do the following:
- If the weak learner selected at iteration k is a regular weak learner (outputs 1 for x<sub>i(k)</sub> greater than its threshold), store the value 255 on location i(k) of array a.
- If the weak learner selected at iteration k is a twin weak learner (outputs −1 for x<sub>i(k)</sub> greater than its threshold), store the value 0 on location i(k) of array a.
Create the array a for each of the 10 binary classifiers and make a plot of the 10 arrays. Comment on what the classifiers are doing to reach their classification decision.

# Results
![image](https://user-images.githubusercontent.com/15370068/161124206-025a7d8e-dc9a-4f58-a050-483885e21750.png)
![image](https://user-images.githubusercontent.com/15370068/161124255-29b2cee9-6cee-4900-ad94-f83f4767589c.png)
![image](https://user-images.githubusercontent.com/15370068/161124293-83554dca-b295-4fa5-b785-ff24b16ec115.png)
![image](https://user-images.githubusercontent.com/15370068/161124334-528d262d-0b1b-4030-86b0-daf7b1217cb2.png)
![image](https://user-images.githubusercontent.com/15370068/161124369-0b7a6f77-1115-417c-b336-97c7df1cbadd.png)
