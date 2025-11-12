# LU Decomposition and Finding Inverse Matrix for Solving Linear Equations

## Purpose of the Project

In many machine learning and numerical computing applications, we often need to solve systems of linear equations, compute matrix inverses, or factor matrices into simpler components for optimization or stability purposes.
This project demonstrates how to:

* Compute the **inverse of a matrix** using the **Gauss–Jordan elimination method**, and
* Perform an **LU Decomposition** (splitting a matrix into a Lower and an Upper triangular matrix) **from scratch**, followed by **NumPy verification**.

Both implementations are done step by step to deepen understanding of how these matrix operations work internally — beyond using built-in library functions.

---

## Part 1: Matrix Inverse via Gauss–Jordan Elimination

### Steps:

1. **Form the Augmented Matrix** – Combine the input square matrix ( A ) with the identity matrix ( I ) → ([A | I]).
2. **Pivot Selection and Row Swapping** – If a pivot element is 0 (or nearly zero), swap with a row below that has a non-zero element.
3. **Normalize the Pivot Row** – Divide the entire pivot row by its pivot to make the pivot equal to 1.
4. **Eliminate All Other Entries** – Use the pivot row to make all other entries in the pivot column zero.
5. **Extract the Inverse** – Once the left half of the augmented matrix becomes the identity, the right half becomes ( A^{-1} ).

---

## Part 2: LU Decomposition from Scratch

LU Decomposition factorizes a matrix ( A ) into two triangular matrices:

[
A = LU
]

where:

* ( L ) is a **Lower Triangular Matrix** (values on and below the diagonal),
* ( U ) is an **Upper Triangular Matrix** (values on and above the diagonal).

### Steps:

1. Initialize ( L ) as an identity matrix and ( U ) as a zero matrix.
2. For each row:

   * Compute upper elements of ( U ) using previously calculated ( L ) and ( U ) values.
   * Compute lower elements of ( L ) using the current column of ( A ) and previously computed results.
3. Stop if a **zero pivot** is encountered (since division by zero is not allowed).

### Code Summary

* **Upper Triangular (U):** Calculated row by row using already known elements.
* **Lower Triangular (L):** Computed column by column using the same logic but normalized by the pivot of ( U ).
* If any pivot in ( U ) is zero, an error is raised.

**Example Output:**

```
LU Decomposition from Scratch:
L (from scratch):
[[ 1.   0.   0. ]
 [ 2.   1.   0. ]
 [ 1.  -1.5  1. ]]
U (from scratch):
[[ 2.   1.   3. ]
 [ 0.   2.   1. ]
 [ 0.   0.   2.5]]
```

---

## Part 3: NumPy Verification ✅

To ensure both the inversion and decomposition are correct, the results are verified using NumPy’s built-in functions:

* **Inverse Check:** `np.linalg.inv(A)`
* **LU Validation:** Verify that ( A ≈ L \times U ) using `np.allclose(A, L @ U)`.

**Example Output:**

```
Inverse A (NumPy):
[[ 3.5 -1.5 -0.5]
 [-2.   1.   0. ]
 [ 0.5 -0.5  0.5]]

L @ U (from scratch):
[[2. 1. 3.]
 [4. 4. 7.]
 [2. 5. 9.]]

Verification (A == L@U): True
```

---

