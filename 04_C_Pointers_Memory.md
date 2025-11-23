# C Programming Tutorial - Part 4: Pointers and Memory

## Table of Contents
1. [Introduction to Pointers](#introduction-to-pointers)
2. [Pointer Basics](#pointer-basics)
3. [Pointer Arithmetic](#pointer-arithmetic)
4. [Pointers and Arrays](#pointers-and-arrays)
5. [Pointers and Functions](#pointers-and-functions)
6. [Dynamic Memory Allocation](#dynamic-memory-allocation)
7. [Multi-level Pointers](#multi-level-pointers)
8. [Common Pointer Pitfalls](#common-pointer-pitfalls)
9. [Practice Exercises](#practice-exercises)

---

## Introduction to Pointers

A **pointer** is a variable that stores the **memory address** of another variable.

### Why Pointers are Critical for CUDA
- **GPU Memory Management**: Allocating and transferring data to/from GPU
- **Efficient Data Access**: Direct memory manipulation
- **Large Data Structures**: Passing arrays without copying
- **Device Pointers**: Special pointers for GPU memory

### Memory Layout

```
Memory Address    Variable    Value
--------------    --------    -----
0x1000           int x       42
0x1004           int *ptr    0x1000  (points to x)
```

---

## Pointer Basics

### Declaring Pointers

```c
int *ptr;        // Pointer to int
float *fptr;     // Pointer to float
char *cptr;      // Pointer to char
double *dptr;    // Pointer to double
```

### Address-of Operator (&)

```c
#include <stdio.h>

int main() {
    int x = 10;
    
    printf("Value of x: %d\n", x);
    printf("Address of x: %p\n", (void*)&x);
    
    return 0;
}
```

### Dereference Operator (*)

```c
#include <stdio.h>

int main() {
    int x = 10;
    int *ptr = &x;  // ptr stores address of x
    
    printf("Value of x: %d\n", x);
    printf("Address of x: %p\n", (void*)&x);
    printf("Value of ptr: %p\n", (void*)ptr);
    printf("Value pointed by ptr: %d\n", *ptr);  // Dereference
    
    // Modify x through pointer
    *ptr = 20;
    printf("New value of x: %d\n", x);
    
    return 0;
}
```

### NULL Pointer

```c
#include <stdio.h>

int main() {
    int *ptr = NULL;  // Initialize to NULL
    
    if (ptr == NULL) {
        printf("Pointer is NULL\n");
    }
    
    // Always check before dereferencing
    if (ptr != NULL) {
        printf("Value: %d\n", *ptr);
    } else {
        printf("Cannot dereference NULL pointer\n");
    }
    
    return 0;
}
```

> [!CAUTION]
> Dereferencing a NULL pointer causes **segmentation fault** (program crash)!

---

## Pointer Arithmetic

Pointers can be incremented, decremented, and compared.

### Basic Arithmetic

```c
#include <stdio.h>

int main() {
    int arr[] = {10, 20, 30, 40, 50};
    int *ptr = arr;  // Points to first element
    
    printf("ptr points to: %d\n", *ptr);
    
    ptr++;  // Move to next element
    printf("After ptr++: %d\n", *ptr);
    
    ptr += 2;  // Move 2 elements forward
    printf("After ptr += 2: %d\n", *ptr);
    
    ptr--;  // Move back one element
    printf("After ptr--: %d\n", *ptr);
    
    return 0;
}
```

### Size Matters

```c
#include <stdio.h>

int main() {
    int arr[] = {1, 2, 3};
    int *iptr = arr;
    
    printf("Address of iptr: %p\n", (void*)iptr);
    iptr++;
    printf("After iptr++: %p\n", (void*)iptr);
    printf("Difference: %ld bytes\n", (char*)iptr - (char*)(iptr-1));
    
    // For int, pointer moves by sizeof(int) bytes
    
    return 0;
}
```

### Pointer Comparison

```c
#include <stdio.h>

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    int *start = arr;
    int *end = arr + 5;
    
    int *ptr = start;
    while (ptr < end) {
        printf("%d ", *ptr);
        ptr++;
    }
    printf("\n");
    
    return 0;
}
```

---

## Pointers and Arrays

### Array Name as Pointer

```c
#include <stdio.h>

int main() {
    int arr[] = {10, 20, 30, 40, 50};
    
    // arr is equivalent to &arr[0]
    printf("arr: %p\n", (void*)arr);
    printf("&arr[0]: %p\n", (void*)&arr[0]);
    
    // Accessing elements
    printf("arr[0] = %d\n", arr[0]);
    printf("*arr = %d\n", *arr);
    printf("*(arr + 1) = %d\n", *(arr + 1));
    printf("arr[2] = %d\n", arr[2]);
    printf("*(arr + 2) = %d\n", *(arr + 2));
    
    return 0;
}
```

### Traversing Array with Pointer

```c
#include <stdio.h>

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    int size = sizeof(arr) / sizeof(arr[0]);
    
    // Method 1: Array indexing
    printf("Method 1: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    
    // Method 2: Pointer arithmetic
    printf("Method 2: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", *(arr + i));
    }
    printf("\n");
    
    // Method 3: Pointer increment
    printf("Method 3: ");
    int *ptr = arr;
    for (int i = 0; i < size; i++) {
        printf("%d ", *ptr);
        ptr++;
    }
    printf("\n");
    
    return 0;
}
```

### Pointer to Array

```c
#include <stdio.h>

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    
    int *ptr = arr;           // Pointer to first element
    int (*arrPtr)[5] = &arr;  // Pointer to entire array
    
    printf("*ptr = %d\n", *ptr);
    printf("(*arrPtr)[0] = %d\n", (*arrPtr)[0]);
    printf("(*arrPtr)[2] = %d\n", (*arrPtr)[2]);
    
    return 0;
}
```

---

## Pointers and Functions

### Passing Pointers to Functions

```c
#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void increment(int *num) {
    (*num)++;
}

int main() {
    int x = 5, y = 10;
    
    printf("Before swap: x = %d, y = %d\n", x, y);
    swap(&x, &y);
    printf("After swap: x = %d, y = %d\n", x, y);
    
    printf("Before increment: x = %d\n", x);
    increment(&x);
    printf("After increment: x = %d\n", x);
    
    return 0;
}
```

### Returning Pointers from Functions

```c
#include <stdio.h>
#include <stdlib.h>

// WRONG: Returning pointer to local variable
int* wrongFunction() {
    int x = 10;
    return &x;  // Dangling pointer!
}

// CORRECT: Returning pointer to dynamically allocated memory
int* correctFunction() {
    int *ptr = (int*)malloc(sizeof(int));
    *ptr = 10;
    return ptr;
}

// CORRECT: Returning pointer to static variable
int* staticFunction() {
    static int x = 10;
    return &x;
}

int main() {
    int *ptr = correctFunction();
    printf("Value: %d\n", *ptr);
    free(ptr);  // Don't forget to free!
    
    return 0;
}
```

### Array as Function Parameter

```c
#include <stdio.h>

// These are equivalent
void printArray1(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void printArray2(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", *(arr + i));
    }
    printf("\n");
}

int main() {
    int numbers[] = {1, 2, 3, 4, 5};
    int size = 5;
    
    printArray1(numbers, size);
    printArray2(numbers, size);
    
    return 0;
}
```

---

## Dynamic Memory Allocation

### malloc() - Memory Allocation

```c
#include <stdio.h>
#include <stdlib.h>

int main() {
    int n;
    
    printf("Enter number of elements: ");
    scanf("%d", &n);
    
    // Allocate memory
    int *arr = (int*)malloc(n * sizeof(int));
    
    // Check if allocation succeeded
    if (arr == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    
    // Use the array
    for (int i = 0; i < n; i++) {
        arr[i] = i + 1;
    }
    
    printf("Array elements: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    
    // Free memory
    free(arr);
    arr = NULL;  // Good practice
    
    return 0;
}
```

### calloc() - Contiguous Allocation

```c
#include <stdio.h>
#include <stdlib.h>

int main() {
    int n = 5;
    
    // Allocates and initializes to zero
    int *arr = (int*)calloc(n, sizeof(int));
    
    if (arr == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    
    printf("Array elements (initialized to 0): ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    
    free(arr);
    
    return 0;
}
```

### realloc() - Reallocation

```c
#include <stdio.h>
#include <stdlib.h>

int main() {
    int *arr = (int*)malloc(3 * sizeof(int));
    
    arr[0] = 1;
    arr[1] = 2;
    arr[2] = 3;
    
    printf("Original array: ");
    for (int i = 0; i < 3; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    
    // Resize to 5 elements
    arr = (int*)realloc(arr, 5 * sizeof(int));
    
    arr[3] = 4;
    arr[4] = 5;
    
    printf("Resized array: ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    
    free(arr);
    
    return 0;
}
```

### Memory Leak Example

```c
#include <stdio.h>
#include <stdlib.h>

void memoryLeak() {
    int *ptr = (int*)malloc(sizeof(int));
    *ptr = 10;
    // Forgot to free(ptr)!
}

void noMemoryLeak() {
    int *ptr = (int*)malloc(sizeof(int));
    *ptr = 10;
    free(ptr);  // Properly freed
}

int main() {
    for (int i = 0; i < 1000000; i++) {
        memoryLeak();  // Leaks memory!
    }
    
    return 0;
}
```

> [!WARNING]
> Always `free()` dynamically allocated memory to avoid memory leaks!

---

## Multi-level Pointers

### Pointer to Pointer

```c
#include <stdio.h>

int main() {
    int x = 10;
    int *ptr = &x;      // Pointer to int
    int **pptr = &ptr;  // Pointer to pointer to int
    
    printf("Value of x: %d\n", x);
    printf("Value using ptr: %d\n", *ptr);
    printf("Value using pptr: %d\n", **pptr);
    
    printf("\nAddress of x: %p\n", (void*)&x);
    printf("Value of ptr: %p\n", (void*)ptr);
    printf("Address of ptr: %p\n", (void*)&ptr);
    printf("Value of pptr: %p\n", (void*)pptr);
    
    // Modify through double pointer
    **pptr = 20;
    printf("\nNew value of x: %d\n", x);
    
    return 0;
}
```

### 2D Array with Pointers

```c
#include <stdio.h>
#include <stdlib.h>

int main() {
    int rows = 3, cols = 4;
    
    // Allocate array of pointers
    int **matrix = (int**)malloc(rows * sizeof(int*));
    
    // Allocate each row
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
    }
    
    // Initialize matrix
    int value = 1;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = value++;
        }
    }
    
    // Print matrix
    printf("Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", matrix[i][j]);
        }
        printf("\n");
    }
    
    // Free memory
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
    
    return 0;
}
```

### Function Pointers

```c
#include <stdio.h>

int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }

int main() {
    // Array of function pointers
    int (*operations[3])(int, int) = {add, subtract, multiply};
    char *names[] = {"Add", "Subtract", "Multiply"};
    
    int a = 10, b = 5;
    
    for (int i = 0; i < 3; i++) {
        printf("%s: %d\n", names[i], operations[i](a, b));
    }
    
    return 0;
}
```

---

## Common Pointer Pitfalls

### 1. Uninitialized Pointer

```c
// WRONG
int *ptr;
*ptr = 10;  // Undefined behavior!

// CORRECT
int *ptr = NULL;
int x;
ptr = &x;
*ptr = 10;
```

### 2. Dangling Pointer

```c
// WRONG
int *ptr = (int*)malloc(sizeof(int));
free(ptr);
*ptr = 10;  // Accessing freed memory!

// CORRECT
int *ptr = (int*)malloc(sizeof(int));
free(ptr);
ptr = NULL;  // Set to NULL after freeing
```

### 3. Memory Leak

```c
// WRONG
void function() {
    int *ptr = (int*)malloc(sizeof(int));
    // Forgot to free!
}

// CORRECT
void function() {
    int *ptr = (int*)malloc(sizeof(int));
    // Use ptr
    free(ptr);
}
```

### 4. Buffer Overflow

```c
// WRONG
int arr[5];
int *ptr = arr;
for (int i = 0; i <= 5; i++) {  // Goes beyond array!
    ptr[i] = i;
}

// CORRECT
int arr[5];
int *ptr = arr;
for (int i = 0; i < 5; i++) {
    ptr[i] = i;
}
```

---

## 📝 Practice Questions (Fill in as you learn)

### Question 1: Pointer Basics
**Q:** What is a pointer?  
**A:** _________________________________

**Q:** What does the `&` operator do?  
**A:** _________________________________

**Q:** What does the `*` operator do when used with pointers?  
**A:** _________________________________

**Q:** What is a NULL pointer?  
**A:** _________________________________

### Question 2: Pointer Arithmetic
**Q:** If `ptr` points to an integer, what does `ptr + 1` point to?  
**A:** _________________________________

**Q:** How many bytes does a pointer move when incremented?  
**A:** _________________________________

**Q:** Can you subtract two pointers?  
**A:** _________________________________

### Question 3: Pointers and Arrays
**Q:** What is the relationship between arrays and pointers?  
**A:** _________________________________

**Q:** What does `arr[i]` translate to using pointer notation?  
**A:** _________________________________

**Q:** Can you change what an array name points to?  
**A:** _________________________________

### Question 4: Dynamic Memory
**Q:** What is the difference between `malloc()` and `calloc()`?  
**A:** _________________________________

**Q:** What does `realloc()` do?  
**A:** _________________________________

**Q:** What happens if you forget to `free()` allocated memory?  
**A:** _________________________________

**Q:** What is a memory leak?  
**A:** _________________________________

### Question 5: Common Issues
**Q:** What is a dangling pointer?  
**A:** _________________________________

**Q:** What causes a segmentation fault?  
**A:** _________________________________

**Q:** What is a buffer overflow?  
**A:** _________________________________

### Question 6: CUDA Preparation
**Q:** Why are pointers critical for CUDA programming?  
**A:** _________________________________

**Q:** What is the difference between host and device pointers in CUDA?  
**A:** _________________________________

---

## Practice Exercises

### Exercise 1: Pointer Basics
Write a program to demonstrate:
- Declaration and initialization of pointers
- Address-of and dereference operators
- Pointer arithmetic

### Exercise 2: Array Reversal
Reverse an array using pointers (two-pointer technique).

```c
void reverseArray(int *arr, int size) {
    // Your code here
}
```

### Exercise 3: String Functions
Implement using pointers:
- `strlen()` - String length
- `strcpy()` - String copy
- `strcmp()` - String compare
- `strcat()` - String concatenation

### Exercise 4: Dynamic Array
Create a dynamic array that:
- Starts with size 5
- Doubles in size when full
- Allows adding and removing elements

### Exercise 5: Matrix Operations
Using dynamic 2D arrays:
- Matrix addition
- Matrix multiplication
- Matrix transpose

### Exercise 6: Linked List
Implement a simple linked list with:
- Insert at beginning
- Insert at end
- Delete node
- Display list

```c
struct Node {
    int data;
    struct Node *next;
};
```

---

## CUDA Preparation Notes

> [!IMPORTANT]
> **Critical Pointer Concepts for CUDA:**
> 
> 1. **Host vs Device Pointers**: Different memory spaces
> 2. **Memory Allocation**: `cudaMalloc()` instead of `malloc()`
> 3. **Memory Transfer**: `cudaMemcpy()` between host and device
> 4. **Pointer Arithmetic**: Same rules apply on GPU
> 5. **Memory Management**: Critical for GPU performance

```c
// CUDA-style memory management (preview)
// Host code
int *h_data = (int*)malloc(size * sizeof(int));      // Host memory
int *d_data;
cudaMalloc(&d_data, size * sizeof(int));             // Device memory
cudaMemcpy(d_data, h_data, size * sizeof(int), 
           cudaMemcpyHostToDevice);                  // Copy to GPU
// ... kernel execution ...
cudaMemcpy(h_data, d_data, size * sizeof(int), 
           cudaMemcpyDeviceToHost);                  // Copy from GPU
cudaFree(d_data);                                     // Free device memory
free(h_data);                                         // Free host memory
```

---

## Key Takeaways

1. **Pointers store memory addresses**
2. **`&` gets address, `*` dereferences**
3. **Pointer arithmetic** depends on data type size
4. **Arrays and pointers** are closely related
5. **Dynamic memory** requires manual management
6. **Always free** allocated memory
7. **Check for NULL** before dereferencing
8. **Avoid dangling pointers** and memory leaks

---

## Next Steps

In the next tutorial, you'll learn about:
- Arrays and strings in detail
- Multi-dimensional arrays
- String manipulation
- Character arrays

> [!TIP]
> Master pointers! They're the foundation of efficient C programming and absolutely essential for CUDA.
