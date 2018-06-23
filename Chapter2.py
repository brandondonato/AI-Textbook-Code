'''
Techniques uesd for preprocessing are:
    Binarization
    Mean Removal
    Scaling
    Normalization
'''
import numpy as np
from sklearn import preprocessing

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
