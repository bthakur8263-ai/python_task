import numpy as np

# ===== ARRAY CREATION =====

# np.array() - Create array from list
print("np.array():")
print(np.array([1, 2, 3]))  
print(np.array([[1, 2], [3, 4]]))  
print(np.array([1, 2, 3], dtype=float))  
print(np.array((1, 2, 3)))  
print(np.array(range(5)))  

# np.zeros() - Create array of zeros
print("\nnp.zeros():")
print(np.zeros(5))  
print(np.zeros((2, 3)))   
print(np.zeros((2, 2), dtype=int))  
print(np.zeros(3, dtype=bool)) 
print(np.zeros((2, 3, 4)))   

# np.ones() - Create array of ones
print("\nnp.ones():")
print(np.ones(4))  
print(np.ones((2, 3)))  
print(np.ones(5, dtype=int))  
print(np.ones((3, 3)))  
print(np.ones((2, 2, 2))) 

# np.arange() - Create array with range
print("\nnp.arange():")
print(np.arange(5))  
print(np.arange(2, 8))  
print(np.arange(0, 10, 2))  
print(np.arange(1, 5, 0.5)) 
print(np.arange(10, 0, -1))  


# np.linspace() - Linear spaced array
print("\nnp.linspace():")
print(np.linspace(0, 10, 5))   
print(np.linspace(1, 100, 4))  
print(np.linspace(0, 1, 11))  
print(np.linspace(5, 15, 3))  
print(np.linspace(-5, 5, 6))   

# np.eye() - Identity matrix
print("\nnp.eye():")
print(np.eye(3))  
print(np.eye(2))  
print(np.eye(4, dtype=int))  
print(np.eye(3, k=1))
print(np.eye(3, 4))  

# np.full() - Fill with value
print("\nnp.full():")
print(np.full(5, 7))  
print(np.full((2, 3), 5))  
print(np.full(4, 3.14))  
print(np.full((3, 3), -1))

# ===== MATHEMATICAL OPERATIONS =====

# np.sum() - Sum elements
print("\nnp.sum():")
arr = np.array([1, 2, 3, 4, 5])
print(np.sum(arr))  
print(np.sum(np.array([[1, 2], [3, 4]])))  
print(np.sum(np.array([[1, 2], [3, 4]]), axis=0))  
print(np.sum(np.array([[1, 2], [3, 4]]), axis=1))  
print(np.sum([10, 20, 30])) 

# np.mean() - Average
print("\nnp.mean():")
print(np.mean([1, 2, 3, 4, 5]))  
print(np.mean(np.array([[1, 2], [3, 4]])))  
print(np.mean(np.array([[1, 2], [3, 4]]), axis=0))  
print(np.mean(np.array([[1, 2], [3, 4]]), axis=1))  
print(np.mean([10, 20, 30, 40]))  

# np.std() - Standard deviation
print("\nnp.std():")
print(np.std([1, 2, 3, 4, 5]))  
print(np.std(np.array([[1, 2], [3, 4]])))  
print(np.std([10, 20, 30]))  
print(np.std(np.array([[1, 2], [3, 4]]), axis=0))  
print(np.std(np.array([[1, 2], [3, 4]]), axis=1))  

# np.min() / np.max() - Minimum and Maximum
print("\nnp.min() / np.max():")
arr = np.array([3, 1, 4, 1, 5])
print(np.min(arr))  
print(np.max(arr))  
print(np.min(np.array([[5, 2], [8, 1]])))  
print(np.max(np.array([[5, 2], [8, 1]])))  
print(np.array([[5, 2], [8, 1]]).max(axis=0))  

# ===== ARRAY MANIPULATION =====

# np.reshape() - Change shape
print("\nnp.reshape():")
arr = np.arange(12)
print(np.reshape(arr, (3, 4)))  
print(np.reshape(arr, (2, 6)))  
print(np.reshape(arr, (4, 3)))  
print(arr.reshape(1, 12))  

# np.flatten() - Flatten to 1D
print("\nnp.flatten():")
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(np.ravel(arr2d))  
print(np.ravel(np.array([[[1, 2], [3, 4]]])))  
print(np.array([[5, 6], [7, 8]]).flatten())  
print(np.ravel(np.eye(3)))  
print(np.ravel(np.arange(20).reshape(4, 5)))  

# np.transpose() / .T - Transpose
print("\nnp.transpose():")
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(np.transpose(arr))  
print(arr.T)  
print(np.array([[1, 2], [3, 4], [5, 6]]).T)  
print(np.array([[7, 8]]).T)  
print(np.diag([1, 2, 3]).T)  

# np.concatenate() - Join arrays
print("\nnp.concatenate():")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.concatenate([a, b]))  
print(np.concatenate([np.array([[1, 2]]), np.array([[3, 4]])], axis=0))  
print(np.concatenate([np.array([[1], [2]]), np.array([[3], [4]])], axis=1))  
print(np.concatenate([[1], [2, 3], [4]]))  
print(np.concatenate([np.arange(3), np.arange(3, 6)]))  

# np.sort() - Sort array
print("\nnp.sort():")
print(np.sort([3, 1, 4, 1, 5]))  
print(np.sort([5, 2, 8, 1, 9]))  
print(np.sort([3, 1, 4, 1, 5])[::-1])  
print(np.sort(np.array([[3, 1], [4, 2]])))  
print(np.sort(np.array([[3, 1], [4, 2]]), axis=0))  

# ===== STATISTICAL FUNCTIONS =====

# np.median() - Median
print("\nnp.median():")
print(np.median([1, 2, 3, 4, 5]))  
print(np.median([1, 2, 3, 4]))  
print(np.median(np.array([[1, 2, 3], [4, 5, 6]])))  
print(np.median([10, 20, 30, 40, 50])) 
print(np.median(np.array([[1, 2], [3, 4], [5, 6]]), axis=0)) 

# np.percentile() - Percentile
print("\nnp.percentile():")
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(np.percentile(data, 25))  
print(np.percentile(data, 50))  
print(np.percentile(data, 75)) 
print(np.percentile(data, 90)) 
print(np.percentile(data, [25, 50, 75]))

# ===== ELEMENT-WISE OPERATIONS =====

# np.sqrt() - Square root
print("\nnp.sqrt():")
print(np.sqrt([1, 4, 9, 16, 25]))  
print(np.sqrt(np.array([[4, 9], [16, 25]])))  
print(np.sqrt([2, 3, 5]))  
print(np.sqrt(100))  
print(np.sqrt(np.arange(1, 6))) 

# np.exp() - Exponential
print("\nnp.exp():")
print(np.exp([0, 1, 2])) 
print(np.exp(np.array([0, 0.5, 1]))) 
print(np.exp([-1, 0, 1]))  
print(np.exp(1)) 
print(np.exp(np.linspace(0, 1, 3))) 

# np.log() - Natural logarithm
print("\nnp.log():")
print(np.log([1, 2.718, 10])) 
print(np.log(np.array([1, np.e, np.e**2]))) 
print(np.log([5, 10, 100]))
print(np.log(np.exp([0, 1, 2]))) 
print(np.log([2, 4, 8]))  

# np.abs() - Absolute value
print("\nnp.abs():")
print(np.abs([-1, -2, 3, -4])) 
print(np.abs(np.array([[-1, 2], [-3, 4]])))
print(np.abs(-5))  
print(np.abs(np.array([-10.5, 20.3, -15.7])))  
print(np.abs(np.linspace(-5, 5, 5))) 

# ===== RANDOM FUNCTIONS =====

# np.random.rand() - Uniform [0, 1)
print("\nnp.random.rand():")
np.random.seed(42)
print(np.random.rand(5)) 
print(np.random.rand(2, 3)) 
print(np.random.rand())
print(np.random.rand(3, 3, 2))  
print(np.random.rand(1, 5))  

# np.random.randint() - Random integers
print("\nnp.random.randint():")
np.random.seed(42)
print(np.random.randint(0, 10, 5))  
print(np.random.randint(1, 7, (2, 3)))  
print(np.random.randint(0, 100))  
print(np.random.randint(50, 100, 4))  
print(np.random.randint(0, 2, 10))  

# np.random.normal() - Normal distribution
print("\nnp.random.normal():")
np.random.seed(42)
print(np.random.normal(0, 1, 5)) 
print(np.random.normal(100, 15, (2, 3)))  
print(np.random.normal(50, 10, 3)) 
print(np.random.normal()) 
print(np.random.normal(0, 1, 4)) 

# ===== LOGICAL OPERATIONS =====

# np.where() - Conditional selection
print("\nnp.where():")
arr = np.array([1, 2, 3, 4, 5])
print(np.where(arr > 3, arr, 0)) 
print(np.where(arr % 2 == 0, 'even', 'odd')) 
print(np.where(arr > 2, arr * 10, arr))  
result = np.where(np.array([[1, 5], [3, 2]]) > 2)
print(result)
print(np.where(arr > 1, arr, 100))  

# np.all() / np.any() - Check conditions
print("\nnp.all() / np.any():")
arr = np.array([True, True, True])
print(np.all(arr))
arr2 = np.array([True, False, True])
print(np.any(arr2))  
print(np.all([1, 2, 3] == [1, 2, 3])) 
print(np.any(np.array([1, 2, 3]) > 2))  
print(np.all(np.array([0, 0, 0]) == 0))  