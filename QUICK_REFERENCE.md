# C/C++ Quick Reference Guide

## C Language Quick Reference

### Data Types
```c
char c = 'A';              // 1 byte
int i = 42;                // 4 bytes
float f = 3.14f;           // 4 bytes
double d = 3.14159;        // 8 bytes
long l = 1000000L;         // 4/8 bytes
```

### Input/Output
```c
printf("Hello %d\n", num);
scanf("%d", &num);
```

### Control Flow
```c
// If-else
if (condition) { } else { }

// Switch
switch (var) {
    case 1: break;
    default: break;
}

// For loop
for (int i = 0; i < n; i++) { }

// While loop
while (condition) { }

// Do-while
do { } while (condition);
```

### Functions
```c
int add(int a, int b) {
    return a + b;
}
```

### Pointers
```c
int x = 10;
int *ptr = &x;     // Pointer to x
*ptr = 20;         // Modify x through pointer
```

### Arrays
```c
int arr[5] = {1, 2, 3, 4, 5};
int matrix[3][3];
```

### Strings
```c
char str[] = "Hello";
strlen(str);
strcpy(dest, src);
strcmp(str1, str2);
```

### Structures
```c
struct Point {
    int x, y;
};

struct Point p = {10, 20};
```

### Dynamic Memory
```c
int *arr = (int*)malloc(n * sizeof(int));
free(arr);
```

---

## C++ Language Quick Reference

### Input/Output
```cpp
cout << "Hello " << num << endl;
cin >> num;
```

### Strings
```cpp
string str = "Hello";
str.length();
str.substr(0, 5);
str + " World";
```

### References
```cpp
int x = 10;
int &ref = x;      // Reference to x
ref = 20;          // Modifies x
```

### Classes
```cpp
class MyClass {
private:
    int data;
public:
    MyClass(int d) : data(d) {}
    void display() { cout << data; }
};
```

### Inheritance
```cpp
class Derived : public Base {
    // Inherits from Base
};
```

### Virtual Functions
```cpp
class Base {
public:
    virtual void func() = 0;  // Pure virtual
};
```

### Templates
```cpp
template <typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}
```

### STL Containers
```cpp
// Vector
vector<int> vec;
vec.push_back(10);
vec[0];
vec.size();

// Map
map<string, int> m;
m["key"] = 10;

// Set
set<int> s;
s.insert(10);
```

### STL Algorithms
```cpp
sort(vec.begin(), vec.end());
find(vec.begin(), vec.end(), value);
reverse(vec.begin(), vec.end());
```

### Smart Pointers
```cpp
auto ptr = make_unique<int>(10);
auto shared = make_shared<int>(20);
```

### Lambda Functions
```cpp
auto lambda = [](int x) { return x * 2; };
```

---

## Common Patterns

### Swap Two Numbers
```c
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}
```

### Array Traversal
```c
for (int i = 0; i < size; i++) {
    printf("%d ", arr[i]);
}
```

### Linked List Node
```c
struct Node {
    int data;
    struct Node *next;
};
```

### Binary Search
```c
int binarySearch(int arr[], int size, int target) {
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
```

### Bubble Sort
```c
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                swap(&arr[j], &arr[j+1]);
            }
        }
    }
}
```

---

## Compilation Commands

### C Compilation
```bash
# Basic
gcc file.c -o program

# With warnings
gcc -Wall file.c -o program

# With optimization
gcc -O2 file.c -o program

# Debug mode
gcc -g file.c -o program

# Link math library
gcc file.c -o program -lm
```

### C++ Compilation
```bash
# Basic
g++ file.cpp -o program

# C++11
g++ -std=c++11 file.cpp -o program

# C++17
g++ -std=c++17 file.cpp -o program

# With warnings
g++ -Wall file.cpp -o program

# With optimization
g++ -O2 file.cpp -o program
```

---

## Common Errors and Solutions

### Segmentation Fault
- **Cause**: Accessing invalid memory
- **Solution**: Check pointers, array bounds

### Memory Leak
- **Cause**: Not freeing allocated memory
- **Solution**: Always `free()` or `delete`

### Undefined Reference
- **Cause**: Missing function definition
- **Solution**: Link all source files

### Dangling Pointer
- **Cause**: Using freed memory
- **Solution**: Set pointer to NULL after free

---

## Best Practices

### C Best Practices
1. Always initialize variables
2. Check malloc() return value
3. Free all allocated memory
4. Use const for read-only parameters
5. Avoid global variables
6. Use meaningful variable names

### C++ Best Practices
1. Use RAII (Resource Acquisition Is Initialization)
2. Prefer smart pointers over raw pointers
3. Use const correctness
4. Avoid using namespace std in headers
5. Use STL containers instead of arrays
6. Make destructors virtual in base classes

---

## Format Specifiers

### C Printf/Scanf
```c
%d    // int
%u    // unsigned int
%ld   // long
%lld  // long long
%f    // float
%lf   // double
%c    // char
%s    // string
%p    // pointer
%x    // hexadecimal
```

---

## Useful Macros

```c
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define ABS(x) ((x) < 0 ? -(x) : (x))
#define SWAP(a,b) { int temp = a; a = b; b = temp; }
```

---

## Memory Layout

```
High Address
+------------------+
|      Stack       |  Local variables, function calls
+------------------+
|        ↓         |
|                  |
|        ↑         |
+------------------+
|      Heap        |  Dynamic allocation (malloc/new)
+------------------+
|  Uninitialized   |  BSS segment
|      Data        |
+------------------+
|   Initialized    |  Global/static variables
|      Data        |
+------------------+
|      Code        |  Program instructions
+------------------+
Low Address
```

---

This quick reference covers the most commonly used features. Refer to the full tutorials for detailed explanations!
