'''
    Techniques uesd for preprocessing are:
        Binarization
        Mean Removal
        Scaling
        Normalization
'''
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
import matplotlib.pyplot as plt


input_data = np.array([[5.1, -2.9, 3.3],
                        [-1.2, 7.8, -6.1],
                        [3.9, 0.4, 2.1],
                        [7.3, -9.9, -4.5]])
'''
    Binarization turns the numerical values into binary values based
    on some threshold. Any values over the threshold are true (1),
    any values less than the threshold are false (0).
'''
data_binarized = preprocessing.Binarizer(threshold=1.0).transform(input_data)
print("\nBinarized Data:\n", data_binarized)

'''
    Here, the mean removal places the mean value very close to 0 and
    makes the standard deviation 1.
'''
print("\nBEFORE\n")
print("Mean =", input_data.mean(axis=0))
print("Standard Deviation =", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("\nAFTER\n")
print("Mean =",data_scaled.mean(axis=0))
print("Standard Deviation =", data_scaled.std(axis=0))

'''
    Scaling the data allows the data to be relative to itself. Largest
    values are set to 1 and all other values will be some fraction of
    the largest value.
'''
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range = (0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin-Max Scaled Data:\n", data_scaled_minmax)

'''
    Normalization is the process used to modify the values in the
    vector we have so they can be compared on a common scale.

    Common forms of Normalization are:
        L1 Normalization - Least Absolute Deviations where the
        sums of the absolute values of each row = 1
        (More Robust, better for no outliers)

        L2 Normalization - Least Squares where the sums of the
        squares of each value = 1
        (Not as robust, good for when outliers are important)
'''

data_normalized_l1 = preprocessing.normalize(input_data, norm="l1")
data_normalized_l2 = preprocessing.normalize(input_data, norm="l2")
print("\nL1 Normalized Data:\n", data_normalized_l1)
print("\nL2 Normalized Data:\n", data_normalized_l2)

'''
    Label Encoding:
        You use labls to classify the information we want to develop
        an algorithm around. sklearn interprets labels as numbers.
        Typically labels are words, so you need to turn the words
        into numbers via Label Encoding. This will allow the algorithms
        to operate on the data.
'''

input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
#Create the label encoder and fit the labels
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
print("\nLabel Mapping\n")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

test_labels = ['green','red','black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded Values =", list(encoded_values))

#   Decoding a set of values using the same encoder
encoded_values = [3,0,4,1]
decoded_list  = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("Decoded Lables =", list(decoded_list))

'''
    Logistic Regression Classifier: Used to explain the relationship
    between input variables and output variables. Input variables are
    independent and output variables are dependent. Dependent variables
    can only take a fixed set of variables where the values correspond
    to the types of classifications.

    Using a sigmoid curve to buld a function with various parameters,
    we estimate the association between independent and dependent
    variables.
'''

X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5],
[5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4],
[0.6, 4.9]])
y = np.array([0,0,0,1,1,1,2,2,2,3,3,3])

classifier = linear_model.LogisticRegression(solver='liblinear', C=100000)
classifier.fit(X,y)

    #Visualize the Classifier Here
min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
mesh_step_size = 0.01
    # Define the mesh grid of X and Y values
x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
    np.arange(min_y, max_y, mesh_step_size))

    # Run the classifier on the mesh grid
output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    # Reshape the output array
output = output.reshape(x_vals.shape)
    # Create a plot
plt.figure()
       # Choose a color scheme for the plot
plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
       # Overlay the training points on the plot
plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black',
   linewidth=1, cmap=plt.cm.Paired)
   # Specify the boundaries of the plot
plt.xlim(x_vals.min(), x_vals.max())
plt.ylim(y_vals.min(), y_vals.max())
       # Specify the ticks on the X and Y axes
plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1),
   1.0)))
plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1),
   1.0)))

plt.show()
#This is a test for a new push
