# C Programming Tutorial - Part 1: Basics

## Table of Contents
1. [Introduction to C](#introduction-to-c)
2. [Setting Up Your Environment](#setting-up-your-environment)
3. [Your First C Program](#your-first-c-program)
4. [Data Types](#data-types)
5. [Variables and Constants](#variables-and-constants)
6. [Operators](#operators)
7. [Input/Output](#inputoutput)
8. [Practice Exercises](#practice-exercises)

---

## Introduction to C

C is a **general-purpose, procedural programming language** developed by Dennis Ritchie in 1972. It's the foundation for many modern languages and is essential for:
- System programming
- Embedded systems
- High-performance computing (including CUDA)
- Operating systems development

### Why Learn C for CUDA?
- CUDA C/C++ extends C/C++ with GPU programming capabilities
- Understanding memory management is crucial for GPU programming
- Pointers and memory allocation are fundamental in CUDA
- C's low-level control translates directly to GPU optimization

---

## Setting Up Your Environment

### Windows
1. Install **MinGW-w64** or **MSVC** (Visual Studio)
2. Add compiler to PATH
3. Use any text editor (VS Code, Notepad++, etc.)

### Compilation Command
```bash
gcc filename.c -o filename.exe
./filename.exe
```

---

## Your First C Program

```c
#include <stdio.h>

int main() {
    printf("Hello, World!\n");
    return 0;
}
```

### Breakdown:
- `#include <stdio.h>`: Preprocessor directive to include standard I/O library
- `int main()`: Entry point of every C program
- `printf()`: Function to print output
- `return 0`: Indicates successful program termination

---

## Data Types

### Basic Data Types

| Type | Size (bytes) | Range | Format Specifier |
|------|--------------|-------|------------------|
| `char` | 1 | -128 to 127 | `%c` |
| `unsigned char` | 1 | 0 to 255 | `%c` |
| `short` | 2 | -32,768 to 32,767 | `%hd` |
| `unsigned short` | 2 | 0 to 65,535 | `%hu` |
| `int` | 4 | -2,147,483,648 to 2,147,483,647 | `%d` |
| `unsigned int` | 4 | 0 to 4,294,967,295 | `%u` |
| `long` | 4/8 | Platform dependent | `%ld` |
| `long long` | 8 | Very large range | `%lld` |
| `float` | 4 | ~7 decimal digits | `%f` |
| `double` | 8 | ~15 decimal digits | `%lf` |
| `long double` | 10/16 | Extended precision | `%Lf` |

### Example:

```c
#include <stdio.h>

int main() {
    char letter = 'A';
    int age = 25;
    float height = 5.9f;
    double pi = 3.14159265359;
    
    printf("Character: %c\n", letter);
    printf("Integer: %d\n", age);
    printf("Float: %.2f\n", height);
    printf("Double: %.10lf\n", pi);
    
    // Size of data types
    printf("\nSize of int: %zu bytes\n", sizeof(int));
    printf("Size of float: %zu bytes\n", sizeof(float));
    printf("Size of double: %zu bytes\n", sizeof(double));
    
    return 0;
}
```

---

## Variables and Constants

### Variables

```c
// Declaration
int number;

// Initialization
number = 10;

// Declaration + Initialization
int count = 5;

// Multiple declarations
int a, b, c;
int x = 1, y = 2, z = 3;
```

### Variable Naming Rules:
- Must start with letter or underscore
- Can contain letters, digits, underscores
- Case-sensitive
- Cannot use C keywords

### Constants

```c
// Using #define (preprocessor)
#define PI 3.14159
#define MAX_SIZE 100

// Using const keyword
const int DAYS_IN_WEEK = 7;
const float GRAVITY = 9.8f;
```

---

## Operators

### Arithmetic Operators

```c
#include <stdio.h>

int main() {
    int a = 10, b = 3;
    
    printf("Addition: %d + %d = %d\n", a, b, a + b);
    printf("Subtraction: %d - %d = %d\n", a, b, a - b);
    printf("Multiplication: %d * %d = %d\n", a, b, a * b);
    printf("Division: %d / %d = %d\n", a, b, a / b);
    printf("Modulus: %d %% %d = %d\n", a, b, a % b);
    
    // Increment/Decrement
    int x = 5;
    printf("\nOriginal x: %d\n", x);
    printf("x++: %d\n", x++);  // Post-increment
    printf("After x++: %d\n", x);
    printf("++x: %d\n", ++x);  // Pre-increment
    
    return 0;
}
```

### Relational Operators

```c
int a = 5, b = 10;

a == b  // Equal to (false)
a != b  // Not equal to (true)
a > b   // Greater than (false)
a < b   // Less than (true)
a >= b  // Greater than or equal to (false)
a <= b  // Less than or equal to (true)
```

### Logical Operators

```c
int x = 1, y = 0;

x && y  // Logical AND (false)
x || y  // Logical OR (true)
!x      // Logical NOT (false)
```

### Bitwise Operators

```c
int a = 5;   // Binary: 0101
int b = 3;   // Binary: 0011

a & b   // AND: 0001 (1)
a | b   // OR: 0111 (7)
a ^ b   // XOR: 0110 (6)
~a      // NOT: 1010 (inverted)
a << 1  // Left shift: 1010 (10)
a >> 1  // Right shift: 0010 (2)
```

### Assignment Operators

```c
int x = 10;

x += 5;  // x = x + 5
x -= 3;  // x = x - 3
x *= 2;  // x = x * 2
x /= 4;  // x = x / 4
x %= 3;  // x = x % 3
```

---

## Input/Output

### Output with printf()

```c
#include <stdio.h>

int main() {
    int age = 25;
    float height = 5.9f;
    char grade = 'A';
    
    // Basic printing
    printf("Age: %d\n", age);
    printf("Height: %.1f\n", height);
    printf("Grade: %c\n", grade);
    
    // Multiple values
    printf("Age: %d, Height: %.1f, Grade: %c\n", age, height, grade);
    
    // Width and precision
    printf("Number: %5d\n", 42);      // Right-aligned, width 5
    printf("Number: %-5d\n", 42);     // Left-aligned, width 5
    printf("Float: %8.2f\n", 3.14);   // Width 8, 2 decimal places
    
    return 0;
}
```

### Input with scanf()

```c
#include <stdio.h>

int main() {
    int age;
    float height;
    char initial;
    
    printf("Enter your age: ");
    scanf("%d", &age);
    
    printf("Enter your height: ");
    scanf("%f", &height);
    
    printf("Enter your initial: ");
    scanf(" %c", &initial);  // Space before %c to consume newline
    
    printf("\nYou entered:\n");
    printf("Age: %d\n", age);
    printf("Height: %.2f\n", height);
    printf("Initial: %c\n", initial);
    
    return 0;
}
```

> [!IMPORTANT]
> Always use `&` (address-of operator) with `scanf()` for basic data types. This gives `scanf()` the memory address where to store the input.

---

## 📝 Practice Questions (Fill in as you learn)

### Question 1: Data Types
**Q:** What is the size of `int` on most modern systems?  
**A:** _________________________________

**Q:** What is the difference between `float` and `double`?  
**A:** _________________________________

**Q:** What does the `sizeof()` operator return?  
**A:** _________________________________

### Question 2: Variables and Constants
**Q:** What is the difference between `#define` and `const`?  
**A:** _________________________________

**Q:** Can you change the value of a constant variable?  
**A:** _________________________________

### Question 3: Operators
**Q:** What is the difference between `++i` and `i++`?  
**A:** _________________________________

**Q:** What does the modulus operator `%` do?  
**A:** _________________________________

**Q:** What is the result of `5 & 3` (bitwise AND)?  
**A:** _________________________________

### Question 4: Input/Output
**Q:** Why do we use `&` with `scanf()` but not with arrays?  
**A:** _________________________________

**Q:** What is the format specifier for printing a pointer address?  
**A:** _________________________________

### Question 5: Conceptual
**Q:** What happens if you don't initialize a variable in C?  
**A:** _________________________________

**Q:** What is the difference between `=` and `==`?  
**A:** _________________________________

---

## Practice Exercises

### Exercise 1: Basic Calculator
Create a program that takes two numbers and an operator (+, -, *, /) and performs the calculation.

```c
#include <stdio.h>

int main() {
    float num1, num2, result;
    char operator;
    
    printf("Enter first number: ");
    scanf("%f", &num1);
    
    printf("Enter operator (+, -, *, /): ");
    scanf(" %c", &operator);
    
    printf("Enter second number: ");
    scanf("%f", &num2);
    
    // Your code here to perform calculation
    
    return 0;
}
```

### Exercise 2: Temperature Converter
Convert Celsius to Fahrenheit and vice versa.
- Formula: F = (C × 9/5) + 32
- Formula: C = (F - 32) × 5/9

### Exercise 3: Area Calculator
Calculate the area of:
- Circle: π × r²
- Rectangle: length × width
- Triangle: (base × height) / 2

### Exercise 4: Swap Two Numbers
Write a program to swap two numbers using a temporary variable.

### Exercise 5: Bitwise Operations
Given two numbers, perform all bitwise operations and display results in both decimal and binary.

---

## Key Takeaways

1. **C is compiled**: Source code → Compiler → Machine code
2. **Strongly typed**: Every variable must have a declared type
3. **Case-sensitive**: `Variable` and `variable` are different
4. **Semicolons matter**: Every statement ends with `;`
5. **Memory awareness**: Understanding data type sizes is crucial
6. **Format specifiers**: Must match the data type in printf/scanf

---

## Next Steps

In the next tutorial, you'll learn about:
- Control flow (if-else, switch)
- Loops (for, while, do-while)
- Break and continue statements
- Nested control structures

> [!TIP]
> Practice writing small programs daily. Understanding these basics is crucial for CUDA programming where you'll manage thousands of threads!
