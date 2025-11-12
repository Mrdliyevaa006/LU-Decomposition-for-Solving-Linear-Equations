 #* Math4AI: Linear Algebra - Programming Assignment 3

import numpy as np

def print_matrix(name, m):
    print(f"{name}:")
    if m is None:
        print("None (Matrix is singular or function not implemented)")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(m)
    print("-" * 30)

A = np.array([
    [2., 1., 3.],
    [4., 4., 7.],
    [2., 5., 9.]
])

print_matrix("Original Matrix A", A)


#* Matrix Inverse via Gauss-Jordan Elimination

def invert_matrix(A):
    m=A.shape[0]  #count of rows
    n=A.shape[1]  #count of columns

    if m!=n: #checking for square matrix
        raise ValueError("Input matrix must be square.")
    
    I_matrix=np.identity(m) #Identity matrix
    augmented=np.hstack((A,I_matrix))
    
    for i in range(m):
        pivot=augmented[i,i]
        if np.isclose(pivot, 0):
            row=None #If the pivot is zero → a non-zero pivot is found from the rows below.
            for a in range(i+1,m):
                if not np.isclose(augmented[a,i],0):
                    row=a
                    break
            if row is None:
                return None
            
            #Swapping rows
            augmented[[i, row]] = augmented[[row, i]]
            pivot = augmented[i, i]

        #Normalizing pivot row
        augmented[i] = augmented[i] / pivot

        #Eliminating rows
        for j in range(m):
            if j!=i:
                augmented[j]-= augmented[j,i] * augmented[i]


    inverse_A = augmented[:, m:]
    return inverse_A

print("--- Part 3.1: Matrix Inverse from Scratch ---")
A_inv_scratch = invert_matrix(A.copy())
print_matrix("Inverse A (from scratch)", A_inv_scratch)        

def lu_decomposition(A):
    A = A.astype(float)
    m=A.shape[0]  
    n=A.shape[1]
    if m != n:
        raise ValueError("Input matrix must be square.")

    L = np.identity(m)
    U = np.zeros((m, m)) #We fill the U matrix with zeros.
    #Then, the elements of the upper triangular matrix will be calculated here.


    #This loop calculates the upper triangular elements in the i-th row of the U matrix.
    for i in range(m):
        for j in range(i, m):
            sum_u = sum(L[i, k] * U[k, j] for k in range(i)) #the sum of already computed values from previous elements
            U[i, j] = A[i, j] - sum_u

        #This loop calculates the lower triangular elements in the i-th column of the L matrix.
        for j in range(i+1, m):
            sum_l = sum(L[j, k] * U[k, i] for k in range(i))
            if np.isclose(U[i, i], 0): #If it is close to zero, the process is stopped (because division is not possible if the pivot is zero)
                raise ValueError("Zero pivot encountered; LU fails without pivoting.")
            L[j, i] = (A[j, i] - sum_l) / U[i, i]

    return L, U

print("LU Decomposition from Scratch:")
L_scratch, U_scratch = lu_decomposition(A.copy())
print_matrix("L (from scratch)", L_scratch)
print_matrix("U (from scratch)", U_scratch)

#*NumPy Verification
print("--- Part 3.3: NumPy Verification ---")
# Matrix inverse
A_inv_numpy = np.linalg.inv(A)
print_matrix("Inverse A (NumPy)", A_inv_numpy)

# LU Verification
product_LU = L_scratch @ U_scratch
print_matrix("L @ U (from scratch)", product_LU)
print_matrix("Original A (for comparison)", A)
print(f"Verification (A == L@U): {np.allclose(A, product_LU)}")






