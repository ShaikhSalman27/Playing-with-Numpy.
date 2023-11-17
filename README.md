# Playing-with-Numpy.

## First Simple Python Variable
w1, w2, w3 = 0.3, 0.2, 0.5

kanto_temp = 73
kanto_rainfall = 67
kanto_humidity = 43

kanto_yield_apples = kanto_temp * w1 + kanto_rainfall * w2 + kanto_humidity * w3
kanto_yield_apples

kanto = [73, 67, 43]
johto = [91, 88, 64]
huenn = [87, 134, 58]
sinnoh = [102, 43, 37]
anova = [69, 96, 70]

wieghts = [w1, w2, w3]

zip(kanto, wieghts)

for item in zip(kanto, wieghts):
    print(item)

kanto

wieghts

for x, w in zip(kanto, wieghts):
    print(x)
    print(w)

result = 0
for x, y in zip(kanto, wieghts):
    result = result + x*y
print(result)

# def function

def crop_yield(region, weights):
    result = 0
    for x,y in zip(region, weights):
        result += x*y
    return result
crop_yield(kanto, wieghts)

crop_yield(johto, wieghts)

crop_yield(anova, wieghts)

def croppo(left, right):
    result = 0
    for a, b in zip(left, right):
        result += a*b
    return result
croppo(huenn, wieghts)

croppo(sinnoh, wieghts)

## Going from Python list to Numpy arrays

import numpy as np

kanto = np.array([73, 67, 43])
kanto

weights = np.array([w1, w2, w3])
weights

type(kanto)

type(weights)

weights[0]

kanto[2]

### Oprating in Numpy arrays

np.dot(kanto, weights)

kanto

weights

kanto * weights

(kanto * weights).sum()

np.sum(kanto * weights)

arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])

arr1 * arr2

arr2.sum()

# Python list
arr1 = list(range(1000000))
arr2 = list(range(1000000, 2000000))

# Numpy arrays
arr1_np = np.array(arr1)
arr2_np = np.array(arr2)

%%time
result = 0 
for x1, x2 in zip(arr1, arr2):
    result += x1*x2
result

%%time
np.dot(arr1_np, arr2_np)

### Multi-dimensinal Numpy arrays

climate_data = np.array([[73, 67, 43],
                        [91, 88, 64],
                        [87, 134, 58],
                        [102, 43, 37],
                        [69, 96, 70]])
climate_data

climate_data.shape

weights

weights.shape

### 3D array

arr3 = np.array([[[11, 12, 13], [13, 14, 15]],
                 [[15, 16, 17], [17, 18, 19.5]]])
arr3

arr3.shape

weights

weights.dtype

climate_data.dtype

arr3.dtype

climate_data

weights

np.matmul(climate_data, weights)

climate_data @ weights

### Arithmatic operation & broadcasting

arr2 = np.array([[1,2,3,4], [5,6,7,8], [9,1,2,3]])
arr3 = np.array([[11,12,13,14], [15,16,17,18], [19,11,12,13]])

arr2 * arr3

arr2 + 3

arr3 - arr2

arr2/2

arr2 % 4

arr2 = np.array([[1,2,3,4], [5,6,7,8], [9,1,2,3]])

arr2.shape

arr4 = np.array([4,5,6,7])

arr4.shape

arr2 + arr4

arr1 = np.array([[[1,2], [3,4], [5,6]],
                 [[7,8], [9,10], [11,12]]])

arr1

arr1.shape

arr2 = np.array([5,6])
arr2.shape

arr1 + arr2

### Comparison Operator

arr1 = np.array([[1,2,3], [3,4,5]])
arr2 = np.array([[2,2,3], [1,2,5]])

arr1 == arr2

arr1 != arr2

arr1 >= arr2

arr1 < arr2

(arr1 == arr2).sum()

np.sum(arr1 == arr2)

### Array indexing & slicing

arr3 = np.array([
    [[11,12,13,14],
     [13,14,15,19]],
    [[15,16,17,21],
     [63,92,36,18]],
    [[98,32,81,23],
    [17,18,19.5,43]]
])

arr3

arr3.shape

arr3[1,1,2]

arr3[0,1,3]

arr3[1:]

arr3[1:, 0:1]

arr3[1:, 0:1, :2]

arr3[1:, 1, 3]

arr3[1:, 1, :3]

arr3[1]

arr3[:2, 1]



### Other ways of creating Numpy arrays

# All Zeros

np.zeros((3,2))

# All ones

np.ones([3,2,3])

# Identity Matrix

np.eye(3)

# Rendom vesctor

np.random.rand(5)

# Rand = Uniform distribution used 
np.random.rand(5,2)

# Randn = Normal distribution used 
np.random.randn(2,3)

np.arange(10, 90, 7)

np.arange(10, 90, 3).reshape(3, 3, 3)

# Equal spaced numbers in a range

np.linspace(3, 27, 6)

np.linspace(3, 27, 9).reshape(3,3)

