# C Programming Tutorial - Part 5: Arrays, Strings, and Structures

## Table of Contents
1. [Arrays](#arrays)
2. [Multi-dimensional Arrays](#multi-dimensional-arrays)
3. [Strings](#strings)
4. [String Functions](#string-functions)
5. [Structures](#structures)
6. [Unions](#unions)
7. [Enumerations](#enumerations)
8. [Typedef](#typedef)
9. [Practice Exercises](#practice-exercises)

---

## Arrays

### Array Declaration and Initialization

```c
#include <stdio.h>

int main() {
    // Declaration
    int arr1[5];
    
    // Declaration with initialization
    int arr2[5] = {1, 2, 3, 4, 5};
    
    // Partial initialization (rest are 0)
    int arr3[5] = {1, 2};  // {1, 2, 0, 0, 0}
    
    // Size inferred from initializer
    int arr4[] = {1, 2, 3, 4, 5};
    
    // Initialize all to zero
    int arr5[5] = {0};
    
    // Print array
    for (int i = 0; i < 5; i++) {
        printf("%d ", arr2[i]);
    }
    printf("\n");
    
    return 0;
}
```

### Array Operations

```c
#include <stdio.h>

int main() {
    int arr[5] = {5, 2, 8, 1, 9};
    int size = sizeof(arr) / sizeof(arr[0]);
    
    // Find maximum
    int max = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    printf("Maximum: %d\n", max);
    
    // Find minimum
    int min = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    printf("Minimum: %d\n", min);
    
    // Calculate sum
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    printf("Sum: %d\n", sum);
    
    // Calculate average
    double avg = (double)sum / size;
    printf("Average: %.2f\n", avg);
    
    return 0;
}
```

### Array Searching

```c
#include <stdio.h>

// Linear search
int linearSearch(int arr[], int size, int target) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == target) {
            return i;  // Return index
        }
    }
    return -1;  // Not found
}

// Binary search (array must be sorted)
int binarySearch(int arr[], int size, int target) {
    int left = 0, right = size - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;  // Not found
}

int main() {
    int arr[] = {1, 3, 5, 7, 9, 11, 13};
    int size = 7;
    
    int index = binarySearch(arr, size, 7);
    if (index != -1) {
        printf("Found at index %d\n", index);
    } else {
        printf("Not found\n");
    }
    
    return 0;
}
```

### Array Sorting (Bubble Sort)

```c
#include <stdio.h>

void bubbleSort(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    int size = 7;
    
    printf("Original array: ");
    printArray(arr, size);
    
    bubbleSort(arr, size);
    
    printf("Sorted array: ");
    printArray(arr, size);
    
    return 0;
}
```

---

## Multi-dimensional Arrays

### 2D Arrays

```c
#include <stdio.h>

int main() {
    // Declaration and initialization
    int matrix[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    
    // Access elements
    printf("Element at [1][2]: %d\n", matrix[1][2]);
    
    // Print matrix
    printf("\nMatrix:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%3d ", matrix[i][j]);
        }
        printf("\n");
    }
    
    return 0;
}
```

### Matrix Operations

```c
#include <stdio.h>

#define ROWS 3
#define COLS 3

void printMatrix(int mat[][COLS], int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%3d ", mat[i][j]);
        }
        printf("\n");
    }
}

void addMatrices(int a[][COLS], int b[][COLS], int result[][COLS], int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < COLS; j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
}

void multiplyMatrices(int a[][COLS], int b[][COLS], int result[][COLS], int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < COLS; j++) {
            result[i][j] = 0;
            for (int k = 0; k < COLS; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void transpose(int mat[][COLS], int result[][ROWS], int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < COLS; j++) {
            result[j][i] = mat[i][j];
        }
    }
}

int main() {
    int a[ROWS][COLS] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int b[ROWS][COLS] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int result[ROWS][COLS];
    
    printf("Matrix A:\n");
    printMatrix(a, ROWS);
    
    printf("\nMatrix B:\n");
    printMatrix(b, ROWS);
    
    addMatrices(a, b, result, ROWS);
    printf("\nA + B:\n");
    printMatrix(result, ROWS);
    
    multiplyMatrices(a, b, result, ROWS);
    printf("\nA * B:\n");
    printMatrix(result, ROWS);
    
    return 0;
}
```

---

## Strings

### String Basics

```c
#include <stdio.h>

int main() {
    // String as character array
    char str1[] = "Hello";
    
    // Explicit initialization
    char str2[] = {'H', 'e', 'l', 'l', 'o', '\0'};
    
    // Pointer to string literal
    char *str3 = "Hello";
    
    // Fixed size
    char str4[20] = "Hello";
    
    printf("str1: %s\n", str1);
    printf("str2: %s\n", str2);
    printf("str3: %s\n", str3);
    printf("str4: %s\n", str4);
    
    // Character by character
    for (int i = 0; str1[i] != '\0'; i++) {
        printf("%c ", str1[i]);
    }
    printf("\n");
    
    return 0;
}
```

> [!IMPORTANT]
> Strings in C are null-terminated (`\0`). Always ensure your string has space for the null terminator!

### String Input/Output

```c
#include <stdio.h>

int main() {
    char name[50];
    char sentence[100];
    
    // Read single word
    printf("Enter your name: ");
    scanf("%s", name);  // No & needed for arrays
    
    // Clear input buffer
    while (getchar() != '\n');
    
    // Read entire line
    printf("Enter a sentence: ");
    fgets(sentence, sizeof(sentence), stdin);
    
    printf("\nName: %s\n", name);
    printf("Sentence: %s", sentence);
    
    return 0;
}
```

---

## String Functions

### Custom String Functions

```c
#include <stdio.h>

// String length
int myStrlen(char *str) {
    int len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

// String copy
void myStrcpy(char *dest, char *src) {
    int i = 0;
    while (src[i] != '\0') {
        dest[i] = src[i];
        i++;
    }
    dest[i] = '\0';
}

// String concatenation
void myStrcat(char *dest, char *src) {
    int i = 0, j = 0;
    
    // Find end of dest
    while (dest[i] != '\0') {
        i++;
    }
    
    // Copy src to end of dest
    while (src[j] != '\0') {
        dest[i] = src[j];
        i++;
        j++;
    }
    dest[i] = '\0';
}

// String comparison
int myStrcmp(char *str1, char *str2) {
    int i = 0;
    while (str1[i] != '\0' && str2[i] != '\0') {
        if (str1[i] != str2[i]) {
            return str1[i] - str2[i];
        }
        i++;
    }
    return str1[i] - str2[i];
}

int main() {
    char str1[50] = "Hello";
    char str2[50] = "World";
    char str3[100];
    
    printf("Length of '%s': %d\n", str1, myStrlen(str1));
    
    myStrcpy(str3, str1);
    printf("Copied string: %s\n", str3);
    
    myStrcat(str3, " ");
    myStrcat(str3, str2);
    printf("Concatenated: %s\n", str3);
    
    int cmp = myStrcmp(str1, str2);
    if (cmp < 0) {
        printf("'%s' comes before '%s'\n", str1, str2);
    } else if (cmp > 0) {
        printf("'%s' comes after '%s'\n", str1, str2);
    } else {
        printf("'%s' equals '%s'\n", str1, str2);
    }
    
    return 0;
}
```

### Standard Library String Functions

```c
#include <stdio.h>
#include <string.h>

int main() {
    char str1[50] = "Hello";
    char str2[50] = "World";
    char str3[100];
    
    // strlen - String length
    printf("Length: %zu\n", strlen(str1));
    
    // strcpy - String copy
    strcpy(str3, str1);
    printf("Copied: %s\n", str3);
    
    // strcat - String concatenation
    strcat(str3, " ");
    strcat(str3, str2);
    printf("Concatenated: %s\n", str3);
    
    // strcmp - String comparison
    if (strcmp(str1, str2) == 0) {
        printf("Strings are equal\n");
    } else {
        printf("Strings are different\n");
    }
    
    // strchr - Find character
    char *pos = strchr(str3, 'W');
    if (pos != NULL) {
        printf("'W' found at position: %ld\n", pos - str3);
    }
    
    // strstr - Find substring
    char *substr = strstr(str3, "World");
    if (substr != NULL) {
        printf("'World' found: %s\n", substr);
    }
    
    return 0;
}
```

---

## Structures

### Basic Structure

```c
#include <stdio.h>

// Define structure
struct Student {
    int id;
    char name[50];
    float gpa;
};

int main() {
    // Declare and initialize
    struct Student s1 = {1, "John Doe", 3.8};
    
    // Access members
    printf("ID: %d\n", s1.id);
    printf("Name: %s\n", s1.name);
    printf("GPA: %.2f\n", s1.gpa);
    
    // Modify members
    s1.gpa = 3.9;
    printf("Updated GPA: %.2f\n", s1.gpa);
    
    return 0;
}
```

### Array of Structures

```c
#include <stdio.h>
#include <string.h>

struct Student {
    int id;
    char name[50];
    float gpa;
};

void printStudent(struct Student s) {
    printf("ID: %d, Name: %s, GPA: %.2f\n", s.id, s.name, s.gpa);
}

int main() {
    struct Student students[3] = {
        {1, "Alice", 3.8},
        {2, "Bob", 3.5},
        {3, "Charlie", 3.9}
    };
    
    printf("All students:\n");
    for (int i = 0; i < 3; i++) {
        printStudent(students[i]);
    }
    
    // Find student with highest GPA
    int maxIndex = 0;
    for (int i = 1; i < 3; i++) {
        if (students[i].gpa > students[maxIndex].gpa) {
            maxIndex = i;
        }
    }
    
    printf("\nStudent with highest GPA:\n");
    printStudent(students[maxIndex]);
    
    return 0;
}
```

### Nested Structures

```c
#include <stdio.h>

struct Date {
    int day;
    int month;
    int year;
};

struct Employee {
    int id;
    char name[50];
    struct Date joinDate;
    float salary;
};

int main() {
    struct Employee emp = {
        101,
        "John Smith",
        {15, 6, 2020},
        50000.0
    };
    
    printf("Employee Details:\n");
    printf("ID: %d\n", emp.id);
    printf("Name: %s\n", emp.name);
    printf("Join Date: %02d/%02d/%d\n", 
           emp.joinDate.day, emp.joinDate.month, emp.joinDate.year);
    printf("Salary: $%.2f\n", emp.salary);
    
    return 0;
}
```

### Pointers to Structures

```c
#include <stdio.h>
#include <stdlib.h>

struct Point {
    int x;
    int y;
};

void printPoint(struct Point *p) {
    // Arrow operator for pointer to structure
    printf("Point: (%d, %d)\n", p->x, p->y);
}

void movePoint(struct Point *p, int dx, int dy) {
    p->x += dx;
    p->y += dy;
}

int main() {
    struct Point p1 = {10, 20};
    
    printPoint(&p1);
    
    movePoint(&p1, 5, -3);
    printPoint(&p1);
    
    // Dynamic allocation
    struct Point *p2 = (struct Point*)malloc(sizeof(struct Point));
    p2->x = 100;
    p2->y = 200;
    
    printPoint(p2);
    
    free(p2);
    
    return 0;
}
```

---

## Unions

Unions allow storing different data types in the same memory location.

```c
#include <stdio.h>

union Data {
    int i;
    float f;
    char str[20];
};

int main() {
    union Data data;
    
    printf("Size of union: %zu bytes\n", sizeof(data));
    
    data.i = 10;
    printf("data.i: %d\n", data.i);
    
    data.f = 3.14;
    printf("data.f: %.2f\n", data.f);
    printf("data.i: %d (corrupted!)\n", data.i);  // Overwritten!
    
    strcpy(data.str, "Hello");
    printf("data.str: %s\n", data.str);
    printf("data.f: %.2f (corrupted!)\n", data.f);  // Overwritten!
    
    return 0;
}
```

> [!WARNING]
> Only one member of a union can hold a value at a time. Writing to one member overwrites others!

---

## Enumerations

```c
#include <stdio.h>

// Define enumeration
enum Day {
    MONDAY,     // 0
    TUESDAY,    // 1
    WEDNESDAY,  // 2
    THURSDAY,   // 3
    FRIDAY,     // 4
    SATURDAY,   // 5
    SUNDAY      // 6
};

enum Color {
    RED = 1,
    GREEN = 2,
    BLUE = 4
};

int main() {
    enum Day today = WEDNESDAY;
    
    printf("Today is day number: %d\n", today);
    
    if (today == WEDNESDAY) {
        printf("It's Wednesday!\n");
    }
    
    enum Color favoriteColor = BLUE;
    printf("Favorite color code: %d\n", favoriteColor);
    
    return 0;
}
```

---

## Typedef

`typedef` creates aliases for data types.

```c
#include <stdio.h>

// Typedef for basic types
typedef unsigned long ulong;
typedef unsigned char byte;

// Typedef for structures
typedef struct {
    int x;
    int y;
} Point;

typedef struct {
    int id;
    char name[50];
    float gpa;
} Student;

// Typedef for function pointers
typedef int (*Operation)(int, int);

int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }

int main() {
    // Use typedef aliases
    ulong bigNumber = 1000000UL;
    byte b = 255;
    
    printf("Big number: %lu\n", bigNumber);
    printf("Byte: %u\n", b);
    
    // Use typedef for structures (no need for 'struct' keyword)
    Point p = {10, 20};
    printf("Point: (%d, %d)\n", p.x, p.y);
    
    Student s = {1, "Alice", 3.8};
    printf("Student: %s, GPA: %.2f\n", s.name, s.gpa);
    
    // Use typedef for function pointers
    Operation op = add;
    printf("5 + 3 = %d\n", op(5, 3));
    
    op = subtract;
    printf("5 - 3 = %d\n", op(5, 3));
    
    return 0;
}
```

---

## 📝 Practice Questions (Fill in as you learn)

### Question 1: Arrays
**Q:** How do you find the size of an array?  
**A:** _________________________________

**Q:** What is the index of the first element in an array?  
**A:** _________________________________

**Q:** Can you change the size of an array after declaration?  
**A:** _________________________________

**Q:** What happens if you access an array out of bounds?  
**A:** _________________________________

### Question 2: Multi-dimensional Arrays
**Q:** How is a 2D array stored in memory?  
**A:** _________________________________

**Q:** What is row-major order?  
**A:** _________________________________

**Q:** How do you pass a 2D array to a function?  
**A:** _________________________________

### Question 3: Strings
**Q:** What is the null terminator in C strings?  
**A:** _________________________________

**Q:** What is the difference between `char str[]` and `char *str`?  
**A:** _________________________________

**Q:** Why do we need `strlen()` when we have `sizeof()`?  
**A:** _________________________________

**Q:** What does `strcpy()` do?  
**A:** _________________________________

### Question 4: Structures
**Q:** What is a structure in C?  
**A:** _________________________________

**Q:** How do you access structure members?  
**A:** _________________________________

**Q:** What is the arrow operator `->` used for?  
**A:** _________________________________

**Q:** Can a structure contain another structure?  
**A:** _________________________________

### Question 5: Unions and Enums
**Q:** What is the difference between a structure and a union?  
**A:** _________________________________

**Q:** How much memory does a union occupy?  
**A:** _________________________________

**Q:** What is an enumeration?  
**A:** _________________________________

### Question 6: Typedef
**Q:** What does `typedef` do?  
**A:** _________________________________

**Q:** Why use `typedef` with structures?  
**A:** _________________________________

---

## Practice Exercises

### Exercise 1: Array Manipulation
Create functions for:
- Rotate array left/right by k positions
- Remove duplicates from sorted array
- Merge two sorted arrays

### Exercise 2: String Utilities
Implement:
- Reverse a string
- Check if string is palindrome
- Count vowels and consonants
- Convert to uppercase/lowercase
- Remove spaces from string

### Exercise 3: Matrix Problems
Solve:
- Rotate matrix 90 degrees
- Find saddle point (min in row, max in column)
- Spiral traversal of matrix

### Exercise 4: Student Management System
Create a system using structures:
- Add student
- Display all students
- Search student by ID
- Find student with highest GPA
- Calculate average GPA

### Exercise 5: Complex Numbers
Create a structure for complex numbers and implement:
- Addition
- Subtraction
- Multiplication
- Division

```c
typedef struct {
    float real;
    float imag;
} Complex;
```

---

## Key Takeaways

1. **Arrays**: Fixed-size, contiguous memory
2. **Strings**: Null-terminated character arrays
3. **2D Arrays**: Array of arrays, row-major order
4. **Structures**: Group related data of different types
5. **Unions**: Share memory among members
6. **Enums**: Named integer constants
7. **Typedef**: Create type aliases for readability

---

## Next Steps

In the next tutorial, you'll learn about:
- File I/O operations
- Reading and writing files
- Binary vs text files
- File positioning

> [!NOTE]
> Structures are crucial for CUDA as they help organize data transferred between CPU and GPU!
