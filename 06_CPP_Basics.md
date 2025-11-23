# C++ Programming Tutorial - Part 1: Introduction and Basics

## Table of Contents
1. [Introduction to C++](#introduction-to-c)
2. [C vs C++](#c-vs-c)
3. [Basic Syntax](#basic-syntax)
4. [Input/Output](#inputoutput)
5. [Namespaces](#namespaces)
6. [References](#references)
7. [Function Overloading](#function-overloading)
8. [Default Arguments](#default-arguments)
9. [Inline Functions](#inline-functions)
10. [Practice Exercises](#practice-exercises)

---

## Introduction to C++

C++ is a **general-purpose, object-oriented programming language** created by Bjarne Stroustrup in 1979 as an extension of C.

### Key Features:
- **Object-Oriented**: Classes, inheritance, polymorphism
- **Generic Programming**: Templates
- **Low-level Memory Manipulation**: Like C
- **Standard Template Library (STL)**: Powerful data structures and algorithms
- **CUDA Support**: CUDA C++ extends C++ for GPU programming

### Why C++ for CUDA?
- CUDA supports both C and C++ syntax
- C++ features like classes and templates work in CUDA
- STL can be used on host (CPU) code
- Better code organization for complex GPU applications

---

## C vs C++

### Major Differences

| Feature | C | C++ |
|---------|---|-----|
| **Paradigm** | Procedural | Multi-paradigm (OOP, Generic) |
| **File Extension** | `.c` | `.cpp`, `.cc`, `.cxx` |
| **Input/Output** | `printf`, `scanf` | `cout`, `cin` |
| **Memory Allocation** | `malloc`, `free` | `new`, `delete` |
| **Function Overloading** | No | Yes |
| **Classes** | No (only structs) | Yes |
| **References** | No | Yes |
| **Templates** | No | Yes |
| **Namespaces** | No | Yes |
| **STL** | No | Yes |

---

## Basic Syntax

### Hello World in C++

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

### Compilation

```bash
g++ hello.cpp -o hello.exe
./hello.exe
```

### Comments

```cpp
// Single-line comment

/*
   Multi-line
   comment
*/

/// Documentation comment (for Doxygen)
```

### Variables and Data Types

```cpp
#include <iostream>

int main() {
    // Basic types (same as C)
    int age = 25;
    float height = 5.9f;
    double pi = 3.14159;
    char grade = 'A';
    bool isPassed = true;  // C++ has native bool type
    
    // C++11: auto keyword (type inference)
    auto x = 10;        // int
    auto y = 3.14;      // double
    auto z = "Hello";   // const char*
    
    std::cout << "Age: " << age << std::endl;
    std::cout << "Height: " << height << std::endl;
    std::cout << "Pi: " << pi << std::endl;
    std::cout << "Grade: " << grade << std::endl;
    std::cout << "Passed: " << isPassed << std::endl;
    
    return 0;
}
```

---

## Input/Output

### cout (Console Output)

```cpp
#include <iostream>
#include <iomanip>

int main() {
    int age = 25;
    double salary = 50000.5;
    
    // Basic output
    std::cout << "Age: " << age << std::endl;
    
    // Multiple values
    std::cout << "Age: " << age << ", Salary: " << salary << std::endl;
    
    // Formatting
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Salary: $" << salary << std::endl;
    
    // Width and alignment
    std::cout << std::setw(10) << "Name" << std::setw(10) << "Age" << std::endl;
    std::cout << std::setw(10) << "John" << std::setw(10) << 25 << std::endl;
    
    return 0;
}
```

### cin (Console Input)

```cpp
#include <iostream>
#include <string>

int main() {
    int age;
    double height;
    std::string name;
    
    std::cout << "Enter your name: ";
    std::cin >> name;  // Reads single word
    
    std::cout << "Enter your age: ";
    std::cin >> age;
    
    std::cout << "Enter your height: ";
    std::cin >> height;
    
    std::cout << "\nHello, " << name << "!" << std::endl;
    std::cout << "Age: " << age << std::endl;
    std::cout << "Height: " << height << std::endl;
    
    return 0;
}
```

### getline for Strings

```cpp
#include <iostream>
#include <string>

int main() {
    std::string fullName;
    
    std::cout << "Enter your full name: ";
    std::getline(std::cin, fullName);
    
    std::cout << "Hello, " << fullName << "!" << std::endl;
    
    return 0;
}
```

---

## Namespaces

Namespaces prevent name conflicts.

### Using Namespace

```cpp
#include <iostream>

// Without using namespace
int main() {
    std::cout << "Hello" << std::endl;
    return 0;
}
```

```cpp
#include <iostream>
using namespace std;

// With using namespace
int main() {
    cout << "Hello" << endl;
    return 0;
}
```

### Custom Namespaces

```cpp
#include <iostream>

namespace Math {
    const double PI = 3.14159;
    
    double square(double x) {
        return x * x;
    }
    
    double cube(double x) {
        return x * x * x;
    }
}

namespace Physics {
    const double GRAVITY = 9.8;
    const double SPEED_OF_LIGHT = 299792458;
}

int main() {
    std::cout << "PI: " << Math::PI << std::endl;
    std::cout << "Square of 5: " << Math::square(5) << std::endl;
    std::cout << "Gravity: " << Physics::GRAVITY << std::endl;
    
    return 0;
}
```

### Namespace Aliases

```cpp
#include <iostream>

namespace VeryLongNamespaceName {
    void function() {
        std::cout << "Function called" << std::endl;
    }
}

int main() {
    namespace VLNN = VeryLongNamespaceName;
    VLNN::function();
    
    return 0;
}
```

> [!TIP]
> Avoid `using namespace std;` in header files to prevent name pollution!

---

## References

References are **aliases** to existing variables.

### Basic References

```cpp
#include <iostream>

int main() {
    int x = 10;
    int &ref = x;  // ref is a reference to x
    
    std::cout << "x: " << x << std::endl;
    std::cout << "ref: " << ref << std::endl;
    
    ref = 20;  // Modifies x
    std::cout << "After ref = 20:" << std::endl;
    std::cout << "x: " << x << std::endl;
    std::cout << "ref: " << ref << std::endl;
    
    return 0;
}
```

### References vs Pointers

```cpp
#include <iostream>

int main() {
    int x = 10;
    
    // Pointer
    int *ptr = &x;
    std::cout << "Using pointer: " << *ptr << std::endl;
    *ptr = 20;
    
    // Reference
    int &ref = x;
    std::cout << "Using reference: " << ref << std::endl;
    ref = 30;
    
    std::cout << "Final x: " << x << std::endl;
    
    return 0;
}
```

| Feature | Pointer | Reference |
|---------|---------|-----------|
| **Syntax** | `int *ptr` | `int &ref` |
| **Null value** | Can be NULL | Cannot be NULL |
| **Reassignment** | Can point to different variables | Cannot be reassigned |
| **Dereferencing** | Requires `*` | Automatic |
| **Address** | Use `&` to get address | Already an alias |

### Pass by Reference

```cpp
#include <iostream>

void swapByPointer(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void swapByReference(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

int main() {
    int x = 5, y = 10;
    
    std::cout << "Before: x = " << x << ", y = " << y << std::endl;
    
    swapByPointer(&x, &y);
    std::cout << "After pointer swap: x = " << x << ", y = " << y << std::endl;
    
    swapByReference(x, y);
    std::cout << "After reference swap: x = " << x << ", y = " << y << std::endl;
    
    return 0;
}
```

---

## Function Overloading

C++ allows multiple functions with the same name but different parameters.

```cpp
#include <iostream>

// Function overloading
int add(int a, int b) {
    return a + b;
}

double add(double a, double b) {
    return a + b;
}

int add(int a, int b, int c) {
    return a + b + c;
}

void print(int x) {
    std::cout << "Integer: " << x << std::endl;
}

void print(double x) {
    std::cout << "Double: " << x << std::endl;
}

void print(const char *str) {
    std::cout << "String: " << str << std::endl;
}

int main() {
    std::cout << "add(5, 3) = " << add(5, 3) << std::endl;
    std::cout << "add(5.5, 3.2) = " << add(5.5, 3.2) << std::endl;
    std::cout << "add(1, 2, 3) = " << add(1, 2, 3) << std::endl;
    
    print(10);
    print(3.14);
    print("Hello");
    
    return 0;
}
```

---

## Default Arguments

```cpp
#include <iostream>

// Default arguments must be rightmost
void greet(std::string name, std::string greeting = "Hello") {
    std::cout << greeting << ", " << name << "!" << std::endl;
}

int power(int base, int exponent = 2) {
    int result = 1;
    for (int i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}

void printInfo(std::string name, int age = 0, std::string city = "Unknown") {
    std::cout << "Name: " << name << std::endl;
    std::cout << "Age: " << age << std::endl;
    std::cout << "City: " << city << std::endl;
}

int main() {
    greet("Alice");                    // Uses default "Hello"
    greet("Bob", "Hi");                // Uses "Hi"
    
    std::cout << "5^2 = " << power(5) << std::endl;        // Uses default 2
    std::cout << "5^3 = " << power(5, 3) << std::endl;     // Uses 3
    
    printInfo("John");
    printInfo("Jane", 25);
    printInfo("Jack", 30, "New York");
    
    return 0;
}
```

---

## Inline Functions

Inline functions suggest the compiler to insert the function code directly at the call site.

```cpp
#include <iostream>

inline int square(int x) {
    return x * x;
}

inline int max(int a, int b) {
    return (a > b) ? a : b;
}

int main() {
    std::cout << "Square of 5: " << square(5) << std::endl;
    std::cout << "Max of 10 and 20: " << max(10, 20) << std::endl;
    
    return 0;
}
```

> [!NOTE]
> `inline` is a suggestion to the compiler. The compiler may ignore it for complex functions.

---

## 📝 Practice Questions (Fill in as you learn)

### Question 1: C vs C++
**Q:** What are the main differences between C and C++?  
**A:** _________________________________

**Q:** What is the file extension for C++ source files?  
**A:** _________________________________

**Q:** Can you compile C code with a C++ compiler?  
**A:** _________________________________

### Question 2: Input/Output
**Q:** What is the difference between `cout` and `printf`?  
**A:** _________________________________

**Q:** What does `endl` do?  
**A:** _________________________________

**Q:** How do you read a full line of input in C++?  
**A:** _________________________________

### Question 3: Namespaces
**Q:** What is a namespace?  
**A:** _________________________________

**Q:** What does `using namespace std;` do?  
**A:** _________________________________

**Q:** Why should you avoid `using namespace std;` in header files?  
**A:** _________________________________

### Question 4: References
**Q:** What is a reference in C++?  
**A:** _________________________________

**Q:** What is the difference between a pointer and a reference?  
**A:** _________________________________

**Q:** Can a reference be NULL?  
**A:** _________________________________

**Q:** Can you reassign a reference?  
**A:** _________________________________

### Question 5: Function Features
**Q:** What is function overloading?  
**A:** _________________________________

**Q:** What are default arguments?  
**A:** _________________________________

**Q:** What does the `inline` keyword suggest?  
**A:** _________________________________

### Question 6: CUDA Relevance
**Q:** Does CUDA support C++ features?  
**A:** _________________________________

**Q:** Why learn C++ for CUDA programming?  
**A:** _________________________________

---

## Practice Exercises

### Exercise 1: Temperature Converter
Create overloaded functions to convert:
- Celsius to Fahrenheit
- Fahrenheit to Celsius
- Celsius to Kelvin
- Kelvin to Celsius

### Exercise 2: Calculator
Implement a calculator using function overloading for:
- `int` operations
- `double` operations
- `long` operations

### Exercise 3: String Utilities
Create functions with default arguments:
- `printLine(char ch = '-', int count = 40)`
- `repeat(string str, int times = 1)`

### Exercise 4: Reference Practice
Write functions using references:
- Swap three numbers
- Find min and max in array (return both via references)
- Sort three numbers

### Exercise 5: Namespace Organization
Create namespaces for:
- Math operations (add, subtract, multiply, divide)
- String operations (uppercase, lowercase, reverse)
- Array operations (sum, average, max, min)

---

## Key Takeaways

1. **C++** extends C with OOP and generic programming
2. **cout/cin** are more type-safe than printf/scanf
3. **Namespaces** prevent name conflicts
4. **References** are safer alternatives to pointers
5. **Function overloading** allows same name, different parameters
6. **Default arguments** provide flexibility
7. **Inline functions** can improve performance

---

## Next Steps

In the next tutorial, you'll learn about:
- Classes and objects
- Constructors and destructors
- Access specifiers
- Member functions
- Static members

> [!IMPORTANT]
> Understanding C++ basics is crucial because CUDA C++ uses these features extensively!
