# Median Calculator
def medianCalculator(array):
    arrayNum = len(array)
    if arrayNum < 1:
        return 0
    else:
        return sum(array) / arrayNum
    
# Standard Deviation
def standardDevitation(array):
    devitation = 0.0
    arrayNum = len(array)
    if arrayNum < 1:
        return 0
    else:
        for num in array:
            devitation += (float(num) - medianCalculator(array)) ** 2
        devitation = (devitation / arrayNum) ** 0.5
        return devitation
    
# Correlation
def correlation(array1, array2):
    array1Num = len(array1)
    array2Num = len(array2)
    if array1Num != array2Num:
        return 0
    else:
        array1Median = medianCalculator(array1)
        array2Median = medianCalculator(array2)
        array1Devitation = standardDevitation(array1)
        array2Devitation = standardDevitation(array2)
        correlation = 0.0
        for i in range(array1Num):
            correlation += (array1[i] - array1Median) * (array2[i] - array2Median)
        correlation = correlation / (array1Num * array1Devitation * array2Devitation)
        return correlation


import random as rd

array1 = [rd.randint(1, 100) for i in range(10)]
array2 = [rd.randint(1, 100) for i in range(10)]

print("Correlation: ", correlation(array1, array2))
print("Standard Deviation of Array1: ", standardDevitation(array1))
print("Standard Deviation of Array2: ", standardDevitation(array2))