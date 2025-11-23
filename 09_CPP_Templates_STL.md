# C++ Programming Tutorial - Part 4: Templates and STL

## Table of Contents
1. [Introduction to Templates](#introduction-to-templates)
2. [Function Templates](#function-templates)
3. [Class Templates](#class-templates)
4. [Standard Template Library (STL)](#standard-template-library-stl)
5. [STL Containers](#stl-containers)
6. [STL Iterators](#stl-iterators)
7. [STL Algorithms](#stl-algorithms)
8. [Smart Pointers](#smart-pointers)
9. [Practice Exercises](#practice-exercises)

---

## Introduction to Templates

Templates enable **generic programming** - writing code that works with any data type.

### Why Templates for CUDA?
- CUDA kernels can be templated
- Type-safe GPU code
- Reusable device functions
- Better performance through compile-time optimization

---

## Function Templates

### Basic Function Template

```cpp
#include <iostream>
using namespace std;

// Function template
template <typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    cout << "Max of 10 and 20: " << maximum(10, 20) << endl;
    cout << "Max of 10.5 and 20.3: " << maximum(10.5, 20.3) << endl;
    cout << "Max of 'a' and 'z': " << maximum('a', 'z') << endl;
    
    return 0;
}
```

### Multiple Template Parameters

```cpp
#include <iostream>
using namespace std;

template <typename T1, typename T2>
void display(T1 a, T2 b) {
    cout << "First: " << a << ", Second: " << b << endl;
}

template <typename T>
T add(T a, T b, T c) {
    return a + b + c;
}

int main() {
    display(10, 3.14);
    display("Hello", 42);
    
    cout << "Sum: " << add(1, 2, 3) << endl;
    cout << "Sum: " << add(1.1, 2.2, 3.3) << endl;
    
    return 0;
}
```

### Template Specialization

```cpp
#include <iostream>
#include <cstring>
using namespace std;

// Generic template
template <typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

// Template specialization for const char*
template <>
const char* maximum<const char*>(const char* a, const char* b) {
    return (strcmp(a, b) > 0) ? a : b;
}

int main() {
    cout << "Max of 10 and 20: " << maximum(10, 20) << endl;
    cout << "Max of 'hello' and 'world': " << maximum("hello", "world") << endl;
    
    return 0;
}
```

---

## Class Templates

### Basic Class Template

```cpp
#include <iostream>
using namespace std;

template <typename T>
class Box {
private:
    T value;
    
public:
    Box(T v) : value(v) {}
    
    T getValue() const {
        return value;
    }
    
    void setValue(T v) {
        value = v;
    }
    
    void display() const {
        cout << "Value: " << value << endl;
    }
};

int main() {
    Box<int> intBox(42);
    Box<double> doubleBox(3.14);
    Box<string> stringBox("Hello");
    
    intBox.display();
    doubleBox.display();
    stringBox.display();
    
    return 0;
}
```

### Template with Multiple Parameters

```cpp
#include <iostream>
using namespace std;

template <typename T1, typename T2>
class Pair {
private:
    T1 first;
    T2 second;
    
public:
    Pair(T1 f, T2 s) : first(f), second(s) {}
    
    T1 getFirst() const { return first; }
    T2 getSecond() const { return second; }
    
    void display() const {
        cout << "(" << first << ", " << second << ")" << endl;
    }
};

int main() {
    Pair<int, double> p1(10, 3.14);
    Pair<string, int> p2("Age", 25);
    
    p1.display();
    p2.display();
    
    return 0;
}
```

### Template Array Class

```cpp
#include <iostream>
using namespace std;

template <typename T, int SIZE>
class Array {
private:
    T data[SIZE];
    
public:
    void set(int index, T value) {
        if (index >= 0 && index < SIZE) {
            data[index] = value;
        }
    }
    
    T get(int index) const {
        if (index >= 0 && index < SIZE) {
            return data[index];
        }
        return T();
    }
    
    int size() const {
        return SIZE;
    }
    
    void display() const {
        for (int i = 0; i < SIZE; i++) {
            cout << data[i] << " ";
        }
        cout << endl;
    }
};

int main() {
    Array<int, 5> intArray;
    for (int i = 0; i < 5; i++) {
        intArray.set(i, i * 10);
    }
    intArray.display();
    
    Array<double, 3> doubleArray;
    doubleArray.set(0, 1.1);
    doubleArray.set(1, 2.2);
    doubleArray.set(2, 3.3);
    doubleArray.display();
    
    return 0;
}
```

---

## Standard Template Library (STL)

STL provides powerful, reusable components:
- **Containers**: Data structures
- **Iterators**: Access elements
- **Algorithms**: Operations on containers
- **Function Objects**: Callable objects

---

## STL Containers

### Vector (Dynamic Array)

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> vec;
    
    // Add elements
    vec.push_back(10);
    vec.push_back(20);
    vec.push_back(30);
    
    // Access elements
    cout << "First element: " << vec[0] << endl;
    cout << "Last element: " << vec.back() << endl;
    
    // Size
    cout << "Size: " << vec.size() << endl;
    
    // Iterate
    cout << "Elements: ";
    for (int i = 0; i < vec.size(); i++) {
        cout << vec[i] << " ";
    }
    cout << endl;
    
    // Range-based for loop (C++11)
    cout << "Elements: ";
    for (int x : vec) {
        cout << x << " ";
    }
    cout << endl;
    
    // Remove last element
    vec.pop_back();
    
    // Insert at position
    vec.insert(vec.begin() + 1, 15);
    
    // Erase element
    vec.erase(vec.begin());
    
    // Clear all
    vec.clear();
    
    return 0;
}
```

### List (Doubly Linked List)

```cpp
#include <iostream>
#include <list>
using namespace std;

int main() {
    list<int> lst;
    
    // Add elements
    lst.push_back(10);
    lst.push_back(20);
    lst.push_front(5);
    
    // Display
    cout << "List: ";
    for (int x : lst) {
        cout << x << " ";
    }
    cout << endl;
    
    // Remove
    lst.remove(10);  // Remove all occurrences of 10
    
    // Sort
    lst.sort();
    
    // Reverse
    lst.reverse();
    
    return 0;
}
```

### Map (Key-Value Pairs)

```cpp
#include <iostream>
#include <map>
using namespace std;

int main() {
    map<string, int> ages;
    
    // Insert elements
    ages["Alice"] = 25;
    ages["Bob"] = 30;
    ages["Charlie"] = 28;
    
    // Access
    cout << "Alice's age: " << ages["Alice"] << endl;
    
    // Check if key exists
    if (ages.find("David") != ages.end()) {
        cout << "David found" << endl;
    } else {
        cout << "David not found" << endl;
    }
    
    // Iterate
    cout << "\nAll ages:" << endl;
    for (auto pair : ages) {
        cout << pair.first << ": " << pair.second << endl;
    }
    
    // Erase
    ages.erase("Bob");
    
    return 0;
}
```

### Set (Unique Elements)

```cpp
#include <iostream>
#include <set>
using namespace std;

int main() {
    set<int> s;
    
    // Insert elements
    s.insert(10);
    s.insert(20);
    s.insert(10);  // Duplicate, won't be added
    s.insert(30);
    
    // Display (automatically sorted)
    cout << "Set: ";
    for (int x : s) {
        cout << x << " ";
    }
    cout << endl;
    
    // Check if element exists
    if (s.find(20) != s.end()) {
        cout << "20 found" << endl;
    }
    
    // Size
    cout << "Size: " << s.size() << endl;
    
    return 0;
}
```

### Stack

```cpp
#include <iostream>
#include <stack>
using namespace std;

int main() {
    stack<int> stk;
    
    // Push elements
    stk.push(10);
    stk.push(20);
    stk.push(30);
    
    // Top element
    cout << "Top: " << stk.top() << endl;
    
    // Pop elements
    while (!stk.empty()) {
        cout << stk.top() << " ";
        stk.pop();
    }
    cout << endl;
    
    return 0;
}
```

### Queue

```cpp
#include <iostream>
#include <queue>
using namespace std;

int main() {
    queue<int> q;
    
    // Enqueue
    q.push(10);
    q.push(20);
    q.push(30);
    
    // Front element
    cout << "Front: " << q.front() << endl;
    
    // Dequeue
    while (!q.empty()) {
        cout << q.front() << " ";
        q.pop();
    }
    cout << endl;
    
    return 0;
}
```

---

## STL Iterators

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> vec = {10, 20, 30, 40, 50};
    
    // Forward iteration
    cout << "Forward: ";
    for (vector<int>::iterator it = vec.begin(); it != vec.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    // Reverse iteration
    cout << "Reverse: ";
    for (vector<int>::reverse_iterator it = vec.rbegin(); it != vec.rend(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    // Auto keyword (C++11)
    cout << "Auto: ";
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    return 0;
}
```

---

## STL Algorithms

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<int> vec = {5, 2, 8, 1, 9, 3};
    
    // Sort
    sort(vec.begin(), vec.end());
    cout << "Sorted: ";
    for (int x : vec) cout << x << " ";
    cout << endl;
    
    // Reverse
    reverse(vec.begin(), vec.end());
    cout << "Reversed: ";
    for (int x : vec) cout << x << " ";
    cout << endl;
    
    // Find
    auto it = find(vec.begin(), vec.end(), 8);
    if (it != vec.end()) {
        cout << "Found 8 at position: " << (it - vec.begin()) << endl;
    }
    
    // Count
    int count = count_if(vec.begin(), vec.end(), [](int x) { return x > 5; });
    cout << "Elements > 5: " << count << endl;
    
    // Min and Max
    cout << "Min: " << *min_element(vec.begin(), vec.end()) << endl;
    cout << "Max: " << *max_element(vec.begin(), vec.end()) << endl;
    
    // Sum
    int sum = accumulate(vec.begin(), vec.end(), 0);
    cout << "Sum: " << sum << endl;
    
    return 0;
}
```

### Lambda Functions (C++11)

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Lambda to check if even
    auto isEven = [](int x) { return x % 2 == 0; };
    
    // Count even numbers
    int evenCount = count_if(vec.begin(), vec.end(), isEven);
    cout << "Even numbers: " << evenCount << endl;
    
    // Transform (square each element)
    transform(vec.begin(), vec.end(), vec.begin(), [](int x) { return x * x; });
    
    cout << "Squared: ";
    for (int x : vec) cout << x << " ";
    cout << endl;
    
    // For each
    for_each(vec.begin(), vec.end(), [](int x) { cout << x << " "; });
    cout << endl;
    
    return 0;
}
```

---

## Smart Pointers

### unique_ptr

```cpp
#include <iostream>
#include <memory>
using namespace std;

class MyClass {
public:
    MyClass() { cout << "Constructor" << endl; }
    ~MyClass() { cout << "Destructor" << endl; }
    void display() { cout << "Hello from MyClass" << endl; }
};

int main() {
    // unique_ptr - exclusive ownership
    unique_ptr<MyClass> ptr1(new MyClass());
    ptr1->display();
    
    // Or use make_unique (C++14)
    auto ptr2 = make_unique<MyClass>();
    ptr2->display();
    
    // unique_ptr<MyClass> ptr3 = ptr1;  // ERROR: cannot copy
    unique_ptr<MyClass> ptr3 = move(ptr1);  // OK: transfer ownership
    
    if (ptr1 == nullptr) {
        cout << "ptr1 is now null" << endl;
    }
    
    return 0;
    // Automatic cleanup
}
```

### shared_ptr

```cpp
#include <iostream>
#include <memory>
using namespace std;

class MyClass {
public:
    MyClass() { cout << "Constructor" << endl; }
    ~MyClass() { cout << "Destructor" << endl; }
};

int main() {
    // shared_ptr - shared ownership
    shared_ptr<MyClass> ptr1 = make_shared<MyClass>();
    cout << "Use count: " << ptr1.use_count() << endl;
    
    {
        shared_ptr<MyClass> ptr2 = ptr1;  // Share ownership
        cout << "Use count: " << ptr1.use_count() << endl;
    }
    
    cout << "Use count after scope: " << ptr1.use_count() << endl;
    
    return 0;
    // Destructor called when last shared_ptr is destroyed
}
```

### weak_ptr

```cpp
#include <iostream>
#include <memory>
using namespace std;

int main() {
    shared_ptr<int> shared = make_shared<int>(42);
    weak_ptr<int> weak = shared;
    
    cout << "Use count: " << shared.use_count() << endl;
    
    // Convert weak_ptr to shared_ptr to access
    if (auto temp = weak.lock()) {
        cout << "Value: " << *temp << endl;
    }
    
    shared.reset();  // Release shared_ptr
    
    if (weak.expired()) {
        cout << "weak_ptr is expired" << endl;
    }
    
    return 0;
}
```

---

## 📝 Practice Questions (Fill in as you learn)

### Question 1: Templates
**Q:** What is a template in C++?  
**A:** _________________________________

**Q:** What is the difference between `typename` and `class` in templates?  
**A:** _________________________________

**Q:** What is template specialization?  
**A:** _________________________________

**Q:** Can templates have non-type parameters?  
**A:** _________________________________

### Question 2: Function Templates
**Q:** How does the compiler generate code from templates?  
**A:** _________________________________

**Q:** What is template type deduction?  
**A:** _________________________________

### Question 3: STL Containers
**Q:** What is the STL?  
**A:** _________________________________

**Q:** What is the difference between `vector` and `array`?  
**A:** _________________________________

**Q:** What is the difference between `vector` and `list`?  
**A:** _________________________________

**Q:** What is the difference between `map` and `unordered_map`?  
**A:** _________________________________

**Q:** What is the difference between `set` and `multiset`?  
**A:** _________________________________

### Question 4: Iterators
**Q:** What is an iterator?  
**A:** _________________________________

**Q:** What are the types of iterators?  
**A:** _________________________________

**Q:** What does `begin()` and `end()` return?  
**A:** _________________________________

### Question 5: Algorithms
**Q:** What does `sort()` do?  
**A:** _________________________________

**Q:** What does `find()` return if element is not found?  
**A:** _________________________________

**Q:** What is a lambda function?  
**A:** _________________________________

### Question 6: Smart Pointers
**Q:** What is a smart pointer?  
**A:** _________________________________

**Q:** What is the difference between `unique_ptr` and `shared_ptr`?  
**A:** _________________________________

**Q:** What is `weak_ptr` used for?  
**A:** _________________________________

**Q:** Why use smart pointers instead of raw pointers?  
**A:** _________________________________

### Question 7: CUDA Relevance
**Q:** Can you use templates in CUDA kernels?  
**A:** _________________________________

**Q:** Can you use STL containers on the GPU?  
**A:** _________________________________

**Q:** How are templates useful for CUDA programming?  
**A:** _________________________________

---

## Practice Exercises

### Exercise 1: Generic Stack
Implement a template-based stack class with:
- `push()`
- `pop()`
- `top()`
- `isEmpty()`
- `size()`

### Exercise 2: Generic Linked List
Create a template linked list with:
- `insertAtBeginning()`
- `insertAtEnd()`
- `deleteNode()`
- `search()`
- `display()`

### Exercise 3: STL Practice
Using STL containers and algorithms:
- Read numbers into a vector
- Remove duplicates
- Sort in descending order
- Find median
- Calculate statistics (mean, mode)

### Exercise 4: Word Frequency Counter
Use `map` to count word frequencies in a text.

### Exercise 5: Priority Queue Simulation
Simulate a task scheduler using `priority_queue`.

### Exercise 6: Smart Pointer Practice
Rewrite a class hierarchy using smart pointers instead of raw pointers.

---

## Complete Example: Generic Data Structure

```cpp
#include <iostream>
#include <memory>
using namespace std;

template <typename T>
class Node {
public:
    T data;
    shared_ptr<Node<T>> next;
    
    Node(T value) : data(value), next(nullptr) {}
};

template <typename T>
class LinkedList {
private:
    shared_ptr<Node<T>> head;
    int count;
    
public:
    LinkedList() : head(nullptr), count(0) {}
    
    void insert(T value) {
        auto newNode = make_shared<Node<T>>(value);
        newNode->next = head;
        head = newNode;
        count++;
    }
    
    void display() {
        auto current = head;
        while (current != nullptr) {
            cout << current->data << " -> ";
            current = current->next;
        }
        cout << "NULL" << endl;
    }
    
    int size() {
        return count;
    }
};

int main() {
    LinkedList<int> intList;
    intList.insert(10);
    intList.insert(20);
    intList.insert(30);
    
    cout << "Integer list: ";
    intList.display();
    cout << "Size: " << intList.size() << endl;
    
    LinkedList<string> stringList;
    stringList.insert("World");
    stringList.insert("Hello");
    
    cout << "\nString list: ";
    stringList.display();
    
    return 0;
}
```

---

## Key Takeaways

1. **Templates** enable generic, reusable code
2. **STL** provides powerful, tested data structures
3. **Iterators** provide uniform access to containers
4. **Algorithms** work with any container via iterators
5. **Smart pointers** automate memory management
6. **Lambda functions** enable inline function objects

---

## CUDA Preparation Summary

> [!IMPORTANT]
> **Key Concepts for CUDA:**
> 
> 1. **Templates**: CUDA kernels can be templated
> 2. **Memory Management**: Understanding pointers is crucial
> 3. **STL on Host**: Use STL for CPU-side data management
> 4. **Smart Pointers**: Manage host memory safely
> 5. **Generic Programming**: Write reusable GPU code
> 6. **Performance**: Template specialization for optimization

---

## Next Steps

You're now ready to start learning CUDA! You have mastered:
- ✓ C fundamentals
- ✓ Pointers and memory management
- ✓ C++ OOP concepts
- ✓ Templates and generic programming
- ✓ STL containers and algorithms

> [!TIP]
> Practice these concepts thoroughly before moving to CUDA. The better you understand C/C++, the easier CUDA will be!
