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
