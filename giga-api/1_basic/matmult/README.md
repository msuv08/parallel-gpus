## Performing Matrix Multiplication using MegaGPU

To perform matrix multiplication using the MegaGPU library, follow these steps:

1. Include the `MegaGPU.h` header file in your C++ code:
   ```cpp
   #include "MegaGPU.h"
   ```

2. Instantiate the MegaGPU object in your code:
   ```cpp
   MegaGPU mega;
   ```

3. Prepare the input matrices:
   - Create vectors to store the elements of matrix A, matrix B, and the resulting matrix C.
   - Initialize the vectors with the desired dimensions and values using the `initializeMatrix` function.

4. Call the `performMatrixMultiplication` function of the MegaGPU object to perform matrix multiplication:
   ```cpp
   mega.performMatrixMultiplication(A.data(), B.data(), C.data(), A_rows, A_cols, B_cols);
   ```
   - `A.data()`: Pointer to the elements of matrix A.
   - `B.data()`: Pointer to the elements of matrix B.
   - `C.data()`: Pointer to the elements of the resulting matrix C.
   - `A_rows`: Number of rows in matrix A.
   - `A_cols`: Number of columns in matrix A.
   - `B_cols`: Number of columns in matrix B.

   The `performMatrixMultiplication` function will perform matrix multiplication using the MegaGPU and store the result in matrix C.

5. After the matrix multiplication, the resulting matrix C will contain the product of matrices A and B.

Note: Make sure to have the MegaGPU library properly linked and the necessary dependencies (CUDA) installed before using the `performMatrixMultiplication` function.

## Makefile

The Makefile provided in the project directory contains the necessary rules to compile the project. It specifies the include and library directories for CUDA, as well as the compiler flags.

To compile the project, simply run `make` in the project directory.

To clean up the compiled files, run `make clean`.