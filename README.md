# gpu_mat_inv
Test of matrix inversion on a GPU, main programm is writen in FORTRAN, GPU kernel and wrapper is in C.

We use partial pivoting, which selects an element with the largest absolute value from a column as a pivot.
