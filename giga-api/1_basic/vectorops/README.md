## Computing Dot Product using MegaGPU

To compute the dot product of two vectors using the MegaGPU library, follow these steps:

1. Include the `MegaGPU.h` header file in your C++ code:
   ```cpp
   #include "MegaGPU.h"
   ```

2. Instantiate the MegaGPU object in your code:
   ```cpp
   MegaGPU mega;
   ```

3. Prepare the input vectors:
   - Create two vectors of the same size to store the elements of vectors a and b.
   - Fill the vectors with the desired values.

4. Create a variable to store the dot product result:
   ```cpp
   float dotProductResult = 0.0f;
   ```

5. Call the `computeDotProduct` function of the MegaGPU object to compute the dot product:
   ```cpp
   mega.computeDotProduct(a.data(), b.data(), dotProductResult, a.size());
   ```
   - `a.data()`: Pointer to the elements of vector a.
   - `b.data()`: Pointer to the elements of vector b.
   - `dotProductResult`: Reference to the variable to store the dot product result.
   - `a.size()`: Size of the vectors (assumed to be the same for both vectors).

   The `computeDotProduct` function will compute the dot product of vectors a and b using the MegaGPU and store the result in the `dotProductResult` variable.

## Computing L2 Norm using MegaGPU

To compute the L2 norm of a vector using the MegaGPU library, follow these steps:

1. Include the `MegaGPU.h` header file in your C++ code:
   ```cpp
   #include "MegaGPU.h"
   ```

2. Instantiate the MegaGPU object in your code:
   ```cpp
   MegaGPU mega;
   ```

3. Prepare the input vector:
   - Create a vector to store the elements of the vector.
   - Fill the vector with the desired values.

4. Create a variable to store the L2 norm result:
   ```cpp
   float l2NormResult = 0.0f;
   ```

5. Call the `computeL2Norm` function of the MegaGPU object to compute the L2 norm:
   ```cpp
   mega.computeL2Norm(a.data(), l2NormResult, a.size());
   ```
   - `a.data()`: Pointer to the elements of the vector.
   - `l2NormResult`: Reference to the variable to store the L2 norm result.
   - `a.size()`: Size of the vector.

   The `computeL2Norm` function will compute the L2 norm of the vector using the MegaGPU and store the result in the `l2NormResult` variable.

Note: Make sure to have the MegaGPU library properly linked and the necessary dependencies (CUDA) installed before using the `computeDotProduct` and `computeL2Norm` functions.

## Makefile

The Makefile provided in the project directory contains the necessary rules to compile the project. It specifies the include and library directories for CUDA, as well as the compiler flags.

To compile the project, simply run `make` in the project directory.

To clean up the compiled files, run `make clean`.