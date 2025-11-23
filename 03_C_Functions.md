# C Programming Tutorial - Part 3: Functions

## Table of Contents
1. [Introduction to Functions](#introduction-to-functions)
2. [Function Declaration and Definition](#function-declaration-and-definition)
3. [Function Parameters](#function-parameters)
4. [Return Values](#return-values)
5. [Recursion](#recursion)
6. [Scope and Lifetime](#scope-and-lifetime)
7. [Storage Classes](#storage-classes)
8. [Practice Exercises](#practice-exercises)

---

## Introduction to Functions

Functions are **reusable blocks of code** that perform specific tasks. They help in:
- **Code reusability**: Write once, use many times
- **Modularity**: Break complex problems into smaller parts
- **Maintainability**: Easier to debug and update
- **Abstraction**: Hide implementation details

### Why Functions Matter for CUDA
- CUDA kernels are special functions that run on GPU
- Understanding function parameters is crucial for passing data to GPU
- Recursion concepts help understand parallel execution
- Scope rules are important for thread-local vs shared memory

---

## Function Declaration and Definition

### Function Syntax

```c
return_type function_name(parameter_list) {
    // Function body
    return value;  // if return_type is not void
}
```

### Example: Basic Function

```c
#include <stdio.h>

// Function declaration (prototype)
void greet();

int main() {
    greet();  // Function call
    return 0;
}

// Function definition
void greet() {
    printf("Hello, World!\n");
}
```

### Function with Parameters

```c
#include <stdio.h>

// Function to add two numbers
int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 3);
    printf("5 + 3 = %d\n", result);
    
    return 0;
}
```

---

## Function Parameters

### Pass by Value

In C, arguments are passed by value (a copy is made).

```c
#include <stdio.h>

void modify(int x) {
    x = 100;
    printf("Inside function: x = %d\n", x);
}

int main() {
    int num = 10;
    printf("Before function: num = %d\n", num);
    
    modify(num);
    
    printf("After function: num = %d\n", num);  // Still 10!
    
    return 0;
}
```

**Output:**
```
Before function: num = 10
Inside function: x = 100
After function: num = 10
```

### Pass by Reference (Using Pointers)

To modify the original variable, pass its address.

```c
#include <stdio.h>

void modify(int *x) {
    *x = 100;
    printf("Inside function: *x = %d\n", *x);
}

int main() {
    int num = 10;
    printf("Before function: num = %d\n", num);
    
    modify(&num);  // Pass address
    
    printf("After function: num = %d\n", num);  // Now 100!
    
    return 0;
}
```

### Multiple Parameters

```c
#include <stdio.h>

// Function to swap two numbers
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main() {
    int x = 5, y = 10;
    
    printf("Before swap: x = %d, y = %d\n", x, y);
    swap(&x, &y);
    printf("After swap: x = %d, y = %d\n", x, y);
    
    return 0;
}
```

### Array Parameters

```c
#include <stdio.h>

// Arrays are always passed by reference
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void modifyArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        arr[i] *= 2;
    }
}

int main() {
    int numbers[] = {1, 2, 3, 4, 5};
    int size = 5;
    
    printf("Original array: ");
    printArray(numbers, size);
    
    modifyArray(numbers, size);
    
    printf("Modified array: ");
    printArray(numbers, size);
    
    return 0;
}
```

---

## Return Values

### Returning Single Value

```c
#include <stdio.h>

int square(int n) {
    return n * n;
}

double average(int a, int b) {
    return (a + b) / 2.0;
}

int main() {
    printf("Square of 5: %d\n", square(5));
    printf("Average of 10 and 15: %.2f\n", average(10, 15));
    
    return 0;
}
```

### Returning Multiple Values (Using Pointers)

```c
#include <stdio.h>

void calculate(int a, int b, int *sum, int *diff, int *prod) {
    *sum = a + b;
    *diff = a - b;
    *prod = a * b;
}

int main() {
    int x = 10, y = 5;
    int sum, diff, prod;
    
    calculate(x, y, &sum, &diff, &prod);
    
    printf("Sum: %d\n", sum);
    printf("Difference: %d\n", diff);
    printf("Product: %d\n", prod);
    
    return 0;
}
```

### Returning Status Codes

```c
#include <stdio.h>

int divide(int a, int b, int *result) {
    if (b == 0) {
        return -1;  // Error code
    }
    *result = a / b;
    return 0;  // Success
}

int main() {
    int result;
    
    if (divide(10, 2, &result) == 0) {
        printf("Result: %d\n", result);
    } else {
        printf("Error: Division by zero!\n");
    }
    
    if (divide(10, 0, &result) == 0) {
        printf("Result: %d\n", result);
    } else {
        printf("Error: Division by zero!\n");
    }
    
    return 0;
}
```

---

## Recursion

A function that calls itself is called a **recursive function**.

### Basic Recursion: Factorial

```c
#include <stdio.h>

int factorial(int n) {
    // Base case
    if (n == 0 || n == 1) {
        return 1;
    }
    // Recursive case
    return n * factorial(n - 1);
}

int main() {
    int num = 5;
    printf("Factorial of %d = %d\n", num, factorial(num));
    
    return 0;
}
```

**How it works:**
```
factorial(5)
= 5 * factorial(4)
= 5 * (4 * factorial(3))
= 5 * (4 * (3 * factorial(2)))
= 5 * (4 * (3 * (2 * factorial(1))))
= 5 * (4 * (3 * (2 * 1)))
= 120
```

### Fibonacci Sequence

```c
#include <stdio.h>

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    printf("Fibonacci sequence (first 10 numbers):\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", fibonacci(i));
    }
    printf("\n");
    
    return 0;
}
```

### Sum of Digits

```c
#include <stdio.h>

int sumOfDigits(int n) {
    if (n == 0) {
        return 0;
    }
    return (n % 10) + sumOfDigits(n / 10);
}

int main() {
    int num = 12345;
    printf("Sum of digits of %d = %d\n", num, sumOfDigits(num));
    
    return 0;
}
```

### Tower of Hanoi

```c
#include <stdio.h>

void towerOfHanoi(int n, char from, char to, char aux) {
    if (n == 1) {
        printf("Move disk 1 from %c to %c\n", from, to);
        return;
    }
    
    towerOfHanoi(n - 1, from, aux, to);
    printf("Move disk %d from %c to %c\n", n, from, to);
    towerOfHanoi(n - 1, aux, to, from);
}

int main() {
    int n = 3;
    printf("Tower of Hanoi with %d disks:\n", n);
    towerOfHanoi(n, 'A', 'C', 'B');
    
    return 0;
}
```

> [!WARNING]
> Recursion can be inefficient and cause stack overflow for large inputs. Always consider iterative alternatives for performance-critical code.

---

## Scope and Lifetime

### Local Variables

```c
#include <stdio.h>

void function1() {
    int x = 10;  // Local to function1
    printf("function1: x = %d\n", x);
}

void function2() {
    int x = 20;  // Different x, local to function2
    printf("function2: x = %d\n", x);
}

int main() {
    int x = 5;  // Local to main
    printf("main: x = %d\n", x);
    
    function1();
    function2();
    
    printf("main: x = %d\n", x);  // Still 5
    
    return 0;
}
```

### Global Variables

```c
#include <stdio.h>

int globalVar = 100;  // Global variable

void modify() {
    globalVar = 200;
    printf("Inside modify: globalVar = %d\n", globalVar);
}

int main() {
    printf("Before modify: globalVar = %d\n", globalVar);
    modify();
    printf("After modify: globalVar = %d\n", globalVar);
    
    return 0;
}
```

### Block Scope

```c
#include <stdio.h>

int main() {
    int x = 10;
    
    {
        int x = 20;  // Different variable, shadows outer x
        printf("Inner block: x = %d\n", x);
    }
    
    printf("Outer block: x = %d\n", x);
    
    return 0;
}
```

---

## Storage Classes

### auto (Default)

```c
void function() {
    auto int x = 10;  // Same as: int x = 10;
    // x is created when function is called
    // x is destroyed when function returns
}
```

### static

```c
#include <stdio.h>

void counter() {
    static int count = 0;  // Initialized only once
    count++;
    printf("Count: %d\n", count);
}

int main() {
    counter();  // Count: 1
    counter();  // Count: 2
    counter();  // Count: 3
    
    return 0;
}
```

### extern

```c
// file1.c
int sharedVar = 100;

// file2.c
extern int sharedVar;  // Declares that sharedVar is defined elsewhere

void printShared() {
    printf("Shared variable: %d\n", sharedVar);
}
```

### register

```c
void function() {
    register int i;  // Hint to store in CPU register for faster access
    
    for (i = 0; i < 1000000; i++) {
        // Fast loop counter
    }
}
```

---

## 📝 Practice Questions (Fill in as you learn)

### Question 1: Function Basics
**Q:** What is the difference between function declaration and definition?  
**A:** _________________________________

**Q:** What does `void` mean as a return type?  
**A:** _________________________________

**Q:** Can a function return multiple values directly?  
**A:** _________________________________

### Question 2: Parameters
**Q:** What is the difference between pass by value and pass by reference?  
**A:** _________________________________

**Q:** Why are arrays always passed by reference?  
**A:** _________________________________

**Q:** How do you modify a variable inside a function?  
**A:** _________________________________

### Question 3: Recursion
**Q:** What is a base case in recursion?  
**A:** _________________________________

**Q:** What happens if you don't have a base case?  
**A:** _________________________________

**Q:** What is tail recursion?  
**A:** _________________________________

### Question 4: Scope and Storage
**Q:** What is the difference between local and global variables?  
**A:** _________________________________

**Q:** What does the `static` keyword do in a function?  
**A:** _________________________________

**Q:** What is the lifetime of a local variable?  
**A:** _________________________________

### Question 5: Advanced Concepts
**Q:** What is a function pointer?  
**A:** _________________________________

**Q:** What is a callback function?  
**A:** _________________________________

**Q:** Why should functions have single responsibility?  
**A:** _________________________________

---

## Practice Exercises

### Exercise 1: Prime Number Function
Write a function to check if a number is prime.

```c
int isPrime(int n) {
    // Your code here
}
```

### Exercise 2: GCD and LCM
Write functions to calculate:
- Greatest Common Divisor (GCD)
- Least Common Multiple (LCM)

### Exercise 3: Power Function
Implement `pow(base, exponent)` using:
1. Iteration
2. Recursion

### Exercise 4: Array Operations
Create functions for:
- Finding maximum element
- Finding minimum element
- Calculating sum
- Calculating average
- Reversing array

### Exercise 5: String Length
Implement your own `strlen()` function using:
1. Iteration
2. Recursion

### Exercise 6: Binary Search
Implement binary search using recursion.

```c
int binarySearch(int arr[], int left, int right, int target) {
    // Your code here
}
```

### Exercise 7: Matrix Operations
Create functions for:
- Matrix addition
- Matrix multiplication
- Matrix transpose

### Exercise 8: Palindrome Checker
Write a recursive function to check if a number is a palindrome.

---

## Advanced Function Concepts

### Function Pointers

```c
#include <stdio.h>

int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }

int main() {
    // Declare function pointer
    int (*operation)(int, int);
    
    operation = add;
    printf("5 + 3 = %d\n", operation(5, 3));
    
    operation = subtract;
    printf("5 - 3 = %d\n", operation(5, 3));
    
    operation = multiply;
    printf("5 * 3 = %d\n", operation(5, 3));
    
    return 0;
}
```

### Callback Functions

```c
#include <stdio.h>

void forEach(int arr[], int size, void (*callback)(int)) {
    for (int i = 0; i < size; i++) {
        callback(arr[i]);
    }
}

void printSquare(int n) {
    printf("%d ", n * n);
}

void printCube(int n) {
    printf("%d ", n * n * n);
}

int main() {
    int numbers[] = {1, 2, 3, 4, 5};
    
    printf("Squares: ");
    forEach(numbers, 5, printSquare);
    printf("\n");
    
    printf("Cubes: ");
    forEach(numbers, 5, printCube);
    printf("\n");
    
    return 0;
}
```

---

## Best Practices

> [!TIP]
> **Function Design Guidelines:**
> 
> 1. **Single Responsibility**: Each function should do one thing well
> 2. **Meaningful Names**: Use descriptive names (e.g., `calculateAverage` not `calc`)
> 3. **Keep It Short**: Aim for functions under 50 lines
> 4. **Limit Parameters**: Ideally 3-4 parameters maximum
> 5. **Avoid Side Effects**: Don't modify global state unnecessarily
> 6. **Document**: Add comments for complex functions

```c
// Good: Clear, single purpose
int calculateCircleArea(double radius) {
    const double PI = 3.14159;
    return PI * radius * radius;
}

// Bad: Multiple responsibilities, unclear name
void doStuff(int x, int y, int z, int *a, int *b, int *c) {
    // Too many parameters, unclear purpose
}
```

---

## CUDA Preparation Notes

> [!IMPORTANT]
> **Concepts Critical for CUDA:**
> 
> 1. **Function Parameters**: CUDA kernels receive parameters similarly
> 2. **Pass by Reference**: GPU memory requires explicit pointer management
> 3. **No Recursion in CUDA**: GPUs don't support recursion well
> 4. **Function Inlining**: Important for GPU performance
> 5. **Scope**: Understanding thread-local vs shared scope

---

## Key Takeaways

1. Functions promote **code reusability** and **modularity**
2. **Pass by value** creates a copy; **pass by reference** uses pointers
3. **Recursion** needs a base case to avoid infinite loops
4. **Scope** determines where variables are accessible
5. **Storage classes** control lifetime and visibility
6. **Function pointers** enable dynamic function calls

---

## Next Steps

In the next tutorial, you'll learn about:
- Pointers in depth
- Dynamic memory allocation
- Pointer arithmetic
- Multi-level pointers

> [!NOTE]
> Pointers are the most important concept for CUDA programming. Master them well!
