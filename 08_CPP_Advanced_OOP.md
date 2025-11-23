# C++ Programming Tutorial - Part 3: Advanced OOP

## Table of Contents
1. [Inheritance](#inheritance)
2. [Types of Inheritance](#types-of-inheritance)
3. [Function Overriding](#function-overriding)
4. [Virtual Functions](#virtual-functions)
5. [Abstract Classes](#abstract-classes)
6. [Polymorphism](#polymorphism)
7. [Operator Overloading](#operator-overloading)
8. [Practice Exercises](#practice-exercises)

---

## Inheritance

Inheritance allows a class to inherit properties and methods from another class.

### Basic Inheritance

```cpp
#include <iostream>
using namespace std;

// Base class
class Animal {
protected:
    string name;
    int age;
    
public:
    Animal(string n, int a) : name(n), age(a) {}
    
    void eat() {
        cout << name << " is eating." << endl;
    }
    
    void sleep() {
        cout << name << " is sleeping." << endl;
    }
    
    void display() {
        cout << "Name: " << name << ", Age: " << age << endl;
    }
};

// Derived class
class Dog : public Animal {
private:
    string breed;
    
public:
    Dog(string n, int a, string b) : Animal(n, a), breed(b) {}
    
    void bark() {
        cout << name << " is barking!" << endl;
    }
    
    void displayDog() {
        display();
        cout << "Breed: " << breed << endl;
    }
};

int main() {
    Dog dog("Buddy", 3, "Golden Retriever");
    
    dog.eat();
    dog.sleep();
    dog.bark();
    dog.displayDog();
    
    return 0;
}
```

### Access Specifiers in Inheritance

```cpp
class Base {
public:
    int publicVar;
protected:
    int protectedVar;
private:
    int privateVar;
};

// Public inheritance
class PublicDerived : public Base {
    // publicVar remains public
    // protectedVar remains protected
    // privateVar is not accessible
};

// Protected inheritance
class ProtectedDerived : protected Base {
    // publicVar becomes protected
    // protectedVar remains protected
    // privateVar is not accessible
};

// Private inheritance
class PrivateDerived : private Base {
    // publicVar becomes private
    // protectedVar becomes private
    // privateVar is not accessible
};
```

---

## Types of Inheritance

### Single Inheritance

```cpp
#include <iostream>
using namespace std;

class Vehicle {
protected:
    string brand;
    
public:
    Vehicle(string b) : brand(b) {}
    
    void honk() {
        cout << "Beep beep!" << endl;
    }
};

class Car : public Vehicle {
private:
    int doors;
    
public:
    Car(string b, int d) : Vehicle(b), doors(d) {}
    
    void display() {
        cout << "Brand: " << brand << ", Doors: " << doors << endl;
    }
};

int main() {
    Car car("Toyota", 4);
    car.honk();
    car.display();
    
    return 0;
}
```

### Multilevel Inheritance

```cpp
#include <iostream>
using namespace std;

class LivingBeing {
protected:
    bool isAlive;
    
public:
    LivingBeing() : isAlive(true) {}
    
    void breathe() {
        cout << "Breathing..." << endl;
    }
};

class Animal : public LivingBeing {
protected:
    string species;
    
public:
    Animal(string s) : species(s) {}
    
    void move() {
        cout << species << " is moving." << endl;
    }
};

class Dog : public Animal {
private:
    string name;
    
public:
    Dog(string n) : Animal("Canine"), name(n) {}
    
    void bark() {
        cout << name << " is barking!" << endl;
    }
};

int main() {
    Dog dog("Max");
    
    dog.breathe();  // From LivingBeing
    dog.move();     // From Animal
    dog.bark();     // From Dog
    
    return 0;
}
```

### Multiple Inheritance

```cpp
#include <iostream>
using namespace std;

class Engine {
protected:
    int horsepower;
    
public:
    Engine(int hp) : horsepower(hp) {}
    
    void startEngine() {
        cout << "Engine started (" << horsepower << " HP)" << endl;
    }
};

class GPS {
protected:
    string location;
    
public:
    GPS(string loc) : location(loc) {}
    
    void navigate() {
        cout << "Navigating to: " << location << endl;
    }
};

class Car : public Engine, public GPS {
private:
    string model;
    
public:
    Car(string m, int hp, string loc) 
        : Engine(hp), GPS(loc), model(m) {}
    
    void display() {
        cout << "Model: " << model << endl;
        startEngine();
        navigate();
    }
};

int main() {
    Car car("Tesla Model S", 670, "San Francisco");
    car.display();
    
    return 0;
}
```

### Hierarchical Inheritance

```cpp
#include <iostream>
using namespace std;

class Shape {
protected:
    string color;
    
public:
    Shape(string c) : color(c) {}
    
    void displayColor() {
        cout << "Color: " << color << endl;
    }
};

class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(string c, double r) : Shape(c), radius(r) {}
    
    double area() {
        return 3.14159 * radius * radius;
    }
};

class Rectangle : public Shape {
private:
    double width, height;
    
public:
    Rectangle(string c, double w, double h) 
        : Shape(c), width(w), height(h) {}
    
    double area() {
        return width * height;
    }
};

int main() {
    Circle circle("Red", 5);
    Rectangle rect("Blue", 10, 5);
    
    circle.displayColor();
    cout << "Circle area: " << circle.area() << endl;
    
    rect.displayColor();
    cout << "Rectangle area: " << rect.area() << endl;
    
    return 0;
}
```

---

## Function Overriding

Derived class can override base class methods.

```cpp
#include <iostream>
using namespace std;

class Animal {
public:
    void sound() {
        cout << "Animal makes a sound" << endl;
    }
    
    void eat() {
        cout << "Animal is eating" << endl;
    }
};

class Dog : public Animal {
public:
    // Override sound()
    void sound() {
        cout << "Dog barks: Woof! Woof!" << endl;
    }
};

class Cat : public Animal {
public:
    // Override sound()
    void sound() {
        cout << "Cat meows: Meow! Meow!" << endl;
    }
};

int main() {
    Animal animal;
    Dog dog;
    Cat cat;
    
    animal.sound();  // Animal makes a sound
    dog.sound();     // Dog barks: Woof! Woof!
    cat.sound();     // Cat meows: Meow! Meow!
    
    return 0;
}
```

---

## Virtual Functions

Virtual functions enable **runtime polymorphism**.

### Basic Virtual Functions

```cpp
#include <iostream>
using namespace std;

class Shape {
public:
    virtual void draw() {
        cout << "Drawing a shape" << endl;
    }
    
    virtual double area() {
        return 0;
    }
};

class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(double r) : radius(r) {}
    
    void draw() override {
        cout << "Drawing a circle" << endl;
    }
    
    double area() override {
        return 3.14159 * radius * radius;
    }
};

class Rectangle : public Shape {
private:
    double width, height;
    
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    
    void draw() override {
        cout << "Drawing a rectangle" << endl;
    }
    
    double area() override {
        return width * height;
    }
};

int main() {
    Shape *shapes[3];
    
    shapes[0] = new Shape();
    shapes[1] = new Circle(5);
    shapes[2] = new Rectangle(10, 5);
    
    for (int i = 0; i < 3; i++) {
        shapes[i]->draw();
        cout << "Area: " << shapes[i]->area() << endl << endl;
    }
    
    // Clean up
    for (int i = 0; i < 3; i++) {
        delete shapes[i];
    }
    
    return 0;
}
```

### Virtual Destructor

```cpp
#include <iostream>
using namespace std;

class Base {
public:
    Base() {
        cout << "Base constructor" << endl;
    }
    
    virtual ~Base() {
        cout << "Base destructor" << endl;
    }
};

class Derived : public Base {
private:
    int *data;
    
public:
    Derived() {
        data = new int[100];
        cout << "Derived constructor" << endl;
    }
    
    ~Derived() {
        delete[] data;
        cout << "Derived destructor" << endl;
    }
};

int main() {
    Base *ptr = new Derived();
    delete ptr;  // Calls both destructors due to virtual
    
    return 0;
}
```

> [!IMPORTANT]
> Always make base class destructors virtual when using inheritance to ensure proper cleanup!

---

## Abstract Classes

Abstract classes contain at least one **pure virtual function** and cannot be instantiated.

```cpp
#include <iostream>
using namespace std;

// Abstract class
class Shape {
public:
    // Pure virtual functions
    virtual void draw() = 0;
    virtual double area() = 0;
    virtual double perimeter() = 0;
    
    // Regular virtual function
    virtual void displayInfo() {
        cout << "This is a shape" << endl;
    }
};

class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(double r) : radius(r) {}
    
    void draw() override {
        cout << "Drawing a circle" << endl;
    }
    
    double area() override {
        return 3.14159 * radius * radius;
    }
    
    double perimeter() override {
        return 2 * 3.14159 * radius;
    }
};

class Rectangle : public Shape {
private:
    double width, height;
    
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    
    void draw() override {
        cout << "Drawing a rectangle" << endl;
    }
    
    double area() override {
        return width * height;
    }
    
    double perimeter() override {
        return 2 * (width + height);
    }
};

int main() {
    // Shape s;  // ERROR: Cannot instantiate abstract class
    
    Shape *shapes[2];
    shapes[0] = new Circle(5);
    shapes[1] = new Rectangle(10, 5);
    
    for (int i = 0; i < 2; i++) {
        shapes[i]->draw();
        cout << "Area: " << shapes[i]->area() << endl;
        cout << "Perimeter: " << shapes[i]->perimeter() << endl << endl;
    }
    
    for (int i = 0; i < 2; i++) {
        delete shapes[i];
    }
    
    return 0;
}
```

---

## Polymorphism

### Compile-time Polymorphism (Function Overloading)

```cpp
#include <iostream>
using namespace std;

class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
    
    double add(double a, double b) {
        return a + b;
    }
    
    int add(int a, int b, int c) {
        return a + b + c;
    }
};

int main() {
    Calculator calc;
    
    cout << calc.add(5, 3) << endl;
    cout << calc.add(5.5, 3.2) << endl;
    cout << calc.add(1, 2, 3) << endl;
    
    return 0;
}
```

### Runtime Polymorphism (Virtual Functions)

```cpp
#include <iostream>
using namespace std;

class Animal {
public:
    virtual void makeSound() {
        cout << "Animal sound" << endl;
    }
};

class Dog : public Animal {
public:
    void makeSound() override {
        cout << "Woof!" << endl;
    }
};

class Cat : public Animal {
public:
    void makeSound() override {
        cout << "Meow!" << endl;
    }
};

void playSound(Animal *animal) {
    animal->makeSound();  // Polymorphic call
}

int main() {
    Animal *animals[3];
    animals[0] = new Animal();
    animals[1] = new Dog();
    animals[2] = new Cat();
    
    for (int i = 0; i < 3; i++) {
        playSound(animals[i]);
    }
    
    for (int i = 0; i < 3; i++) {
        delete animals[i];
    }
    
    return 0;
}
```

---

## Operator Overloading

### Arithmetic Operators

```cpp
#include <iostream>
using namespace std;

class Complex {
private:
    double real, imag;
    
public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}
    
    // Overload + operator
    Complex operator+(const Complex &c) {
        return Complex(real + c.real, imag + c.imag);
    }
    
    // Overload - operator
    Complex operator-(const Complex &c) {
        return Complex(real - c.real, imag - c.imag);
    }
    
    // Overload * operator
    Complex operator*(const Complex &c) {
        return Complex(
            real * c.real - imag * c.imag,
            real * c.imag + imag * c.real
        );
    }
    
    void display() {
        cout << real;
        if (imag >= 0) cout << " + " << imag << "i";
        else cout << " - " << -imag << "i";
        cout << endl;
    }
};

int main() {
    Complex c1(3, 4);
    Complex c2(1, 2);
    
    Complex c3 = c1 + c2;
    Complex c4 = c1 - c2;
    Complex c5 = c1 * c2;
    
    cout << "c1: "; c1.display();
    cout << "c2: "; c2.display();
    cout << "c1 + c2: "; c3.display();
    cout << "c1 - c2: "; c4.display();
    cout << "c1 * c2: "; c5.display();
    
    return 0;
}
```

### Comparison Operators

```cpp
#include <iostream>
using namespace std;

class Point {
private:
    int x, y;
    
public:
    Point(int xVal, int yVal) : x(xVal), y(yVal) {}
    
    // Overload == operator
    bool operator==(const Point &p) {
        return (x == p.x && y == p.y);
    }
    
    // Overload != operator
    bool operator!=(const Point &p) {
        return !(*this == p);
    }
    
    void display() {
        cout << "(" << x << ", " << y << ")" << endl;
    }
};

int main() {
    Point p1(10, 20);
    Point p2(10, 20);
    Point p3(5, 15);
    
    if (p1 == p2) {
        cout << "p1 and p2 are equal" << endl;
    }
    
    if (p1 != p3) {
        cout << "p1 and p3 are not equal" << endl;
    }
    
    return 0;
}
```

### Stream Operators

```cpp
#include <iostream>
using namespace std;

class Point {
private:
    int x, y;
    
public:
    Point(int xVal = 0, int yVal = 0) : x(xVal), y(yVal) {}
    
    // Friend functions for stream operators
    friend ostream& operator<<(ostream &out, const Point &p);
    friend istream& operator>>(istream &in, Point &p);
};

// Overload << operator
ostream& operator<<(ostream &out, const Point &p) {
    out << "(" << p.x << ", " << p.y << ")";
    return out;
}

// Overload >> operator
istream& operator>>(istream &in, Point &p) {
    cout << "Enter x: ";
    in >> p.x;
    cout << "Enter y: ";
    in >> p.y;
    return in;
}

int main() {
    Point p1(10, 20);
    cout << "p1: " << p1 << endl;
    
    Point p2;
    cin >> p2;
    cout << "p2: " << p2 << endl;
    
    return 0;
}
```

### Increment/Decrement Operators

```cpp
#include <iostream>
using namespace std;

class Counter {
private:
    int count;
    
public:
    Counter(int c = 0) : count(c) {}
    
    // Prefix increment
    Counter& operator++() {
        count++;
        return *this;
    }
    
    // Postfix increment
    Counter operator++(int) {
        Counter temp = *this;
        count++;
        return temp;
    }
    
    int getCount() const {
        return count;
    }
};

int main() {
    Counter c(5);
    
    cout << "Initial: " << c.getCount() << endl;
    
    ++c;
    cout << "After ++c: " << c.getCount() << endl;
    
    c++;
    cout << "After c++: " << c.getCount() << endl;
    
    return 0;
}
```

---

## 📝 Practice Questions (Fill in as you learn)

### Question 1: Inheritance
**Q:** What is inheritance?  
**A:** _________________________________

**Q:** What is a base class and derived class?  
**A:** _________________________________

**Q:** What are the types of inheritance?  
**A:** _________________________________

**Q:** What is multiple inheritance?  
**A:** _________________________________

### Question 2: Access in Inheritance
**Q:** What happens to private members in inheritance?  
**A:** _________________________________

**Q:** What is the difference between public, protected, and private inheritance?  
**A:** _________________________________

### Question 3: Polymorphism
**Q:** What is polymorphism?  
**A:** _________________________________

**Q:** What is the difference between compile-time and runtime polymorphism?  
**A:** _________________________________

**Q:** What is function overriding?  
**A:** _________________________________

### Question 4: Virtual Functions
**Q:** What is a virtual function?  
**A:** _________________________________

**Q:** What is a pure virtual function?  
**A:** _________________________________

**Q:** Why should destructors be virtual in base classes?  
**A:** _________________________________

**Q:** What is a virtual table (vtable)?  
**A:** _________________________________

### Question 5: Abstract Classes
**Q:** What is an abstract class?  
**A:** _________________________________

**Q:** Can you instantiate an abstract class?  
**A:** _________________________________

**Q:** What is an interface in C++?  
**A:** _________________________________

### Question 6: Operator Overloading
**Q:** What is operator overloading?  
**A:** _________________________________

**Q:** Can you overload all operators?  
**A:** _________________________________

**Q:** What is the difference between prefix and postfix increment operators?  
**A:** _________________________________

**Q:** Why are stream operators (`<<`, `>>`) usually friend functions?  
**A:** _________________________________

---

## Practice Exercises

### Exercise 1: Shape Hierarchy
Create an abstract `Shape` class and derive:
- `Circle`
- `Rectangle`
- `Triangle`

Each should implement `area()` and `perimeter()`.

### Exercise 2: Employee Management
Create a base `Employee` class and derive:
- `Manager` (with bonus)
- `Developer` (with programming language)
- `Designer` (with design tool)

Implement virtual `calculateSalary()`.

### Exercise 3: Vector Class
Create a `Vector` class with operator overloading:
- `+`, `-` (addition, subtraction)
- `*` (dot product)
- `==`, `!=` (comparison)
- `<<`, `>>` (stream I/O)

### Exercise 4: Fraction Class
Create a `Fraction` class with:
- Numerator and denominator
- Overload: `+`, `-`, `*`, `/`
- Simplify fractions
- Overload `<<` for display

### Exercise 5: Matrix Class
Create a `Matrix` class with:
- Dynamic 2D array
- Overload: `+`, `-`, `*`
- Transpose
- Determinant (for 2x2 and 3x3)

---

## Key Takeaways

1. **Inheritance** promotes code reuse
2. **Virtual functions** enable runtime polymorphism
3. **Abstract classes** define interfaces
4. **Operator overloading** makes classes more intuitive
5. **Polymorphism** allows treating derived classes uniformly
6. Always use **virtual destructors** in base classes

---

## Next Steps

In the next tutorial, you'll learn about:
- Templates (function and class templates)
- Standard Template Library (STL)
- Containers, iterators, algorithms
- Smart pointers

> [!NOTE]
> Understanding polymorphism and templates is crucial for advanced CUDA programming patterns!
