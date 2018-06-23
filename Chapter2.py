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
'''Binarization turns the numerical values into binary values based
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
