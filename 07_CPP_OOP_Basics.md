# C++ Programming Tutorial - Part 2: Object-Oriented Programming

## Table of Contents
1. [Introduction to OOP](#introduction-to-oop)
2. [Classes and Objects](#classes-and-objects)
3. [Constructors and Destructors](#constructors-and-destructors)
4. [Access Specifiers](#access-specifiers)
5. [Member Functions](#member-functions)
6. [Static Members](#static-members)
7. [Friend Functions](#friend-functions)
8. [this Pointer](#this-pointer)
9. [Practice Exercises](#practice-exercises)

---

## Introduction to OOP

**Object-Oriented Programming (OOP)** is a programming paradigm based on objects that contain data and methods.

### Four Pillars of OOP:
1. **Encapsulation**: Bundling data and methods together
2. **Abstraction**: Hiding complex implementation details
3. **Inheritance**: Creating new classes from existing ones
4. **Polymorphism**: Same interface, different implementations

### Why OOP for CUDA?
- Organize GPU code into reusable classes
- Encapsulate device memory management
- Create abstractions for kernels and data transfers
- Better code maintainability for complex applications

---

## Classes and Objects

### Basic Class

```cpp
#include <iostream>
using namespace std;

class Rectangle {
public:
    int width;
    int height;
    
    int area() {
        return width * height;
    }
    
    int perimeter() {
        return 2 * (width + height);
    }
};

int main() {
    Rectangle rect;
    rect.width = 10;
    rect.height = 5;
    
    cout << "Area: " << rect.area() << endl;
    cout << "Perimeter: " << rect.perimeter() << endl;
    
    return 0;
}
```

### Class with Methods Outside

```cpp
#include <iostream>
using namespace std;

class Circle {
private:
    double radius;
    
public:
    void setRadius(double r);
    double getRadius();
    double area();
    double circumference();
};

// Method definitions outside class
void Circle::setRadius(double r) {
    radius = r;
}

double Circle::getRadius() {
    return radius;
}

double Circle::area() {
    return 3.14159 * radius * radius;
}

double Circle::circumference() {
    return 2 * 3.14159 * radius;
}

int main() {
    Circle c;
    c.setRadius(5.0);
    
    cout << "Radius: " << c.getRadius() << endl;
    cout << "Area: " << c.area() << endl;
    cout << "Circumference: " << c.circumference() << endl;
    
    return 0;
}
```

---

## Constructors and Destructors

### Default Constructor

```cpp
#include <iostream>
using namespace std;

class Student {
private:
    string name;
    int age;
    
public:
    // Default constructor
    Student() {
        name = "Unknown";
        age = 0;
        cout << "Default constructor called" << endl;
    }
    
    void display() {
        cout << "Name: " << name << ", Age: " << age << endl;
    }
};

int main() {
    Student s;
    s.display();
    
    return 0;
}
```

### Parameterized Constructor

```cpp
#include <iostream>
using namespace std;

class Student {
private:
    string name;
    int age;
    float gpa;
    
public:
    // Parameterized constructor
    Student(string n, int a, float g) {
        name = n;
        age = a;
        gpa = g;
        cout << "Parameterized constructor called" << endl;
    }
    
    void display() {
        cout << "Name: " << name << endl;
        cout << "Age: " << age << endl;
        cout << "GPA: " << gpa << endl;
    }
};

int main() {
    Student s1("Alice", 20, 3.8);
    s1.display();
    
    return 0;
}
```

### Constructor Overloading

```cpp
#include <iostream>
using namespace std;

class Point {
private:
    int x, y;
    
public:
    // Default constructor
    Point() {
        x = 0;
        y = 0;
    }
    
    // Parameterized constructor
    Point(int xVal, int yVal) {
        x = xVal;
        y = yVal;
    }
    
    // Copy constructor
    Point(const Point &p) {
        x = p.x;
        y = p.y;
        cout << "Copy constructor called" << endl;
    }
    
    void display() {
        cout << "(" << x << ", " << y << ")" << endl;
    }
};

int main() {
    Point p1;              // Default constructor
    Point p2(10, 20);      // Parameterized constructor
    Point p3 = p2;         // Copy constructor
    
    cout << "p1: "; p1.display();
    cout << "p2: "; p2.display();
    cout << "p3: "; p3.display();
    
    return 0;
}
```

### Member Initializer List

```cpp
#include <iostream>
using namespace std;

class Rectangle {
private:
    const int width;   // const member
    const int height;  // const member
    
public:
    // Member initializer list (required for const members)
    Rectangle(int w, int h) : width(w), height(h) {
        cout << "Rectangle created: " << width << "x" << height << endl;
    }
    
    int area() const {
        return width * height;
    }
};

int main() {
    Rectangle rect(10, 5);
    cout << "Area: " << rect.area() << endl;
    
    return 0;
}
```

### Destructor

```cpp
#include <iostream>
using namespace std;

class Array {
private:
    int *data;
    int size;
    
public:
    // Constructor
    Array(int s) {
        size = s;
        data = new int[size];
        cout << "Array of size " << size << " created" << endl;
    }
    
    // Destructor
    ~Array() {
        delete[] data;
        cout << "Array destroyed" << endl;
    }
    
    void set(int index, int value) {
        if (index >= 0 && index < size) {
            data[index] = value;
        }
    }
    
    int get(int index) {
        if (index >= 0 && index < size) {
            return data[index];
        }
        return -1;
    }
};

int main() {
    Array arr(5);
    
    for (int i = 0; i < 5; i++) {
        arr.set(i, i * 10);
    }
    
    for (int i = 0; i < 5; i++) {
        cout << arr.get(i) << " ";
    }
    cout << endl;
    
    return 0;
    // Destructor automatically called here
}
```

> [!IMPORTANT]
> Destructors are crucial for cleaning up resources (memory, file handles, etc.). They're automatically called when an object goes out of scope.

---

## Access Specifiers

### public, private, protected

```cpp
#include <iostream>
using namespace std;

class BankAccount {
private:
    string accountNumber;
    double balance;
    
protected:
    string ownerName;
    
public:
    // Constructor
    BankAccount(string acc, string name, double bal) {
        accountNumber = acc;
        ownerName = name;
        balance = bal;
    }
    
    // Public methods to access private data
    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            cout << "Deposited: $" << amount << endl;
        }
    }
    
    void withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            cout << "Withdrawn: $" << amount << endl;
        } else {
            cout << "Insufficient funds!" << endl;
        }
    }
    
    double getBalance() const {
        return balance;
    }
    
    void displayInfo() const {
        cout << "Account: " << accountNumber << endl;
        cout << "Owner: " << ownerName << endl;
        cout << "Balance: $" << balance << endl;
    }
};

int main() {
    BankAccount account("123456", "John Doe", 1000.0);
    
    account.displayInfo();
    
    account.deposit(500);
    account.withdraw(200);
    
    cout << "\nCurrent balance: $" << account.getBalance() << endl;
    
    // account.balance = 10000;  // ERROR: private member
    
    return 0;
}
```

| Specifier | Class | Derived Class | Outside Class |
|-----------|-------|---------------|---------------|
| **public** | ✓ | ✓ | ✓ |
| **protected** | ✓ | ✓ | ✗ |
| **private** | ✓ | ✗ | ✗ |

---

## Member Functions

### Const Member Functions

```cpp
#include <iostream>
using namespace std;

class Point {
private:
    int x, y;
    
public:
    Point(int xVal, int yVal) : x(xVal), y(yVal) {}
    
    // Const member function (doesn't modify object)
    void display() const {
        cout << "(" << x << ", " << y << ")" << endl;
        // x = 10;  // ERROR: cannot modify in const function
    }
    
    int getX() const { return x; }
    int getY() const { return y; }
    
    // Non-const member function
    void move(int dx, int dy) {
        x += dx;
        y += dy;
    }
};

int main() {
    Point p(10, 20);
    
    p.display();
    p.move(5, -3);
    p.display();
    
    const Point cp(100, 200);
    cp.display();  // OK: const function
    // cp.move(1, 1);  // ERROR: non-const function on const object
    
    return 0;
}
```

### Inline Member Functions

```cpp
#include <iostream>
using namespace std;

class Math {
public:
    // Inline member functions
    inline int square(int x) {
        return x * x;
    }
    
    inline int cube(int x) {
        return x * x * x;
    }
    
    // Functions defined inside class are automatically inline
    int add(int a, int b) {
        return a + b;
    }
};

int main() {
    Math m;
    
    cout << "Square of 5: " << m.square(5) << endl;
    cout << "Cube of 3: " << m.cube(3) << endl;
    cout << "5 + 3 = " << m.add(5, 3) << endl;
    
    return 0;
}
```

---

## Static Members

### Static Data Members

```cpp
#include <iostream>
using namespace std;

class Counter {
private:
    static int count;  // Static member declaration
    int id;
    
public:
    Counter() {
        count++;
        id = count;
        cout << "Object " << id << " created" << endl;
    }
    
    ~Counter() {
        cout << "Object " << id << " destroyed" << endl;
        count--;
    }
    
    static int getCount() {
        return count;
    }
};

// Static member definition (required)
int Counter::count = 0;

int main() {
    cout << "Initial count: " << Counter::getCount() << endl;
    
    Counter c1;
    cout << "Count: " << Counter::getCount() << endl;
    
    {
        Counter c2, c3;
        cout << "Count: " << Counter::getCount() << endl;
    }
    
    cout << "Count after scope: " << Counter::getCount() << endl;
    
    return 0;
}
```

### Static Member Functions

```cpp
#include <iostream>
using namespace std;

class Math {
public:
    static const double PI;
    
    static double square(double x) {
        return x * x;
    }
    
    static double circleArea(double radius) {
        return PI * radius * radius;
    }
    
    static int max(int a, int b) {
        return (a > b) ? a : b;
    }
};

const double Math::PI = 3.14159;

int main() {
    // Call static functions without creating object
    cout << "Square of 5: " << Math::square(5) << endl;
    cout << "Circle area (r=3): " << Math::circleArea(3) << endl;
    cout << "Max of 10 and 20: " << Math::max(10, 20) << endl;
    cout << "PI: " << Math::PI << endl;
    
    return 0;
}
```

---

## Friend Functions

Friend functions can access private and protected members of a class.

```cpp
#include <iostream>
using namespace std;

class Box {
private:
    double width;
    double height;
    
public:
    Box(double w, double h) : width(w), height(h) {}
    
    // Friend function declaration
    friend double calculateArea(const Box &b);
    friend void displayBox(const Box &b);
};

// Friend function definition
double calculateArea(const Box &b) {
    return b.width * b.height;  // Can access private members
}

void displayBox(const Box &b) {
    cout << "Box: " << b.width << " x " << b.height << endl;
}

int main() {
    Box box(10, 5);
    
    displayBox(box);
    cout << "Area: " << calculateArea(box) << endl;
    
    return 0;
}
```

### Friend Classes

```cpp
#include <iostream>
using namespace std;

class Engine {
private:
    int horsepower;
    
public:
    Engine(int hp) : horsepower(hp) {}
    
    friend class Car;  // Car can access Engine's private members
};

class Car {
private:
    string model;
    Engine engine;
    
public:
    Car(string m, int hp) : model(m), engine(hp) {}
    
    void display() {
        cout << "Model: " << model << endl;
        cout << "Horsepower: " << engine.horsepower << endl;  // Access private member
    }
};

int main() {
    Car car("Tesla Model S", 670);
    car.display();
    
    return 0;
}
```

---

## this Pointer

`this` is a pointer to the current object.

```cpp
#include <iostream>
using namespace std;

class Person {
private:
    string name;
    int age;
    
public:
    Person(string name, int age) {
        // Use 'this' to distinguish between parameter and member
        this->name = name;
        this->age = age;
    }
    
    void display() {
        cout << "Name: " << this->name << endl;
        cout << "Age: " << this->age << endl;
    }
    
    // Return reference to current object for chaining
    Person& setName(string name) {
        this->name = name;
        return *this;
    }
    
    Person& setAge(int age) {
        this->age = age;
        return *this;
    }
    
    bool isOlderThan(const Person &other) {
        return this->age > other.age;
    }
};

int main() {
    Person p1("Alice", 25);
    p1.display();
    
    // Method chaining
    Person p2("Bob", 30);
    p2.setName("Robert").setAge(35);
    p2.display();
    
    if (p2.isOlderThan(p1)) {
        cout << "p2 is older than p1" << endl;
    }
    
    return 0;
}
```

---

## 📝 Practice Questions (Fill in as you learn)

### Question 1: OOP Concepts
**Q:** What are the four pillars of OOP?  
**A:** _________________________________

**Q:** What is encapsulation?  
**A:** _________________________________

**Q:** What is abstraction?  
**A:** _________________________________

### Question 2: Classes and Objects
**Q:** What is the difference between a class and an object?  
**A:** _________________________________

**Q:** What is a member variable?  
**A:** _________________________________

**Q:** What is a member function?  
**A:** _________________________________

### Question 3: Constructors and Destructors
**Q:** What is a constructor?  
**A:** _________________________________

**Q:** What is a destructor?  
**A:** _________________________________

**Q:** When is a destructor called?  
**A:** _________________________________

**Q:** What is a copy constructor?  
**A:** _________________________________

**Q:** What is a member initializer list?  
**A:** _________________________________

### Question 4: Access Specifiers
**Q:** What are the three access specifiers in C++?  
**A:** _________________________________

**Q:** What is the default access specifier for a class?  
**A:** _________________________________

**Q:** What is the difference between `private` and `protected`?  
**A:** _________________________________

### Question 5: Special Members
**Q:** What is a static member variable?  
**A:** _________________________________

**Q:** What is a static member function?  
**A:** _________________________________

**Q:** What is a friend function?  
**A:** _________________________________

**Q:** What is the `this` pointer?  
**A:** _________________________________

### Question 6: Const Correctness
**Q:** What is a const member function?  
**A:** _________________________________

**Q:** Can a const member function modify member variables?  
**A:** _________________________________

---

## Practice Exercises

### Exercise 1: Complex Number Class
Create a `Complex` class with:
- Real and imaginary parts
- Constructors
- Methods: add, subtract, multiply, display
- Overload operators (next tutorial)

### Exercise 2: Bank Account System
Create a `BankAccount` class with:
- Private: account number, balance
- Public: deposit, withdraw, transfer, display
- Static: total number of accounts, total money in bank

### Exercise 3: Student Management
Create a `Student` class with:
- Private: name, ID, grades array
- Methods: add grade, calculate average, display
- Static: count total students

### Exercise 4: Vector2D Class
Create a 2D vector class with:
- x, y coordinates
- Methods: magnitude, normalize, dot product
- Friend function for cross product

### Exercise 5: Time Class
Create a `Time` class with:
- Hours, minutes, seconds
- Methods: add time, subtract time, compare
- Display in 12-hour and 24-hour format

---

## Advanced Example: String Class

```cpp
#include <iostream>
#include <cstring>
using namespace std;

class String {
private:
    char *str;
    int length;
    
public:
    // Default constructor
    String() {
        length = 0;
        str = new char[1];
        str[0] = '\0';
    }
    
    // Parameterized constructor
    String(const char *s) {
        length = strlen(s);
        str = new char[length + 1];
        strcpy(str, s);
    }
    
    // Copy constructor
    String(const String &s) {
        length = s.length;
        str = new char[length + 1];
        strcpy(str, s.str);
    }
    
    // Destructor
    ~String() {
        delete[] str;
    }
    
    int getLength() const {
        return length;
    }
    
    void display() const {
        cout << str << endl;
    }
    
    void concatenate(const String &s) {
        char *temp = new char[length + s.length + 1];
        strcpy(temp, str);
        strcat(temp, s.str);
        delete[] str;
        str = temp;
        length += s.length;
    }
};

int main() {
    String s1("Hello");
    String s2(" World");
    
    cout << "s1: "; s1.display();
    cout << "s2: "; s2.display();
    
    s1.concatenate(s2);
    cout << "After concatenation: "; s1.display();
    cout << "Length: " << s1.getLength() << endl;
    
    return 0;
}
```

---

## Key Takeaways

1. **Classes** encapsulate data and functions
2. **Constructors** initialize objects
3. **Destructors** clean up resources
4. **Access specifiers** control visibility
5. **Static members** are shared across all objects
6. **Friend functions** can access private members
7. **this pointer** refers to current object

---

## Next Steps

In the next tutorial, you'll learn about:
- Inheritance
- Types of inheritance
- Function overriding
- Virtual functions
- Abstract classes

> [!NOTE]
> OOP concepts are essential for organizing complex CUDA applications with multiple kernels and data structures!
