// Generic Data Structures - Demonstrates C++ templates
#include <iostream>
#include <memory>
using namespace std;

// Template Stack
template <typename T>
class Stack {
private:
    struct Node {
        T data;
        shared_ptr<Node> next;
        Node(T val) : data(val), next(nullptr) {}
    };
    
    shared_ptr<Node> top;
    int count;
    
public:
    Stack() : top(nullptr), count(0) {}
    
    void push(T value) {
        auto newNode = make_shared<Node>(value);
        newNode->next = top;
        top = newNode;
        count++;
    }
    
    T pop() {
        if (isEmpty()) {
            throw runtime_error("Stack is empty!");
        }
        T value = top->data;
        top = top->next;
        count--;
        return value;
    }
    
    T peek() const {
        if (isEmpty()) {
            throw runtime_error("Stack is empty!");
        }
        return top->data;
    }
    
    bool isEmpty() const {
        return top == nullptr;
    }
    
    int size() const {
        return count;
    }
    
    void display() const {
        auto current = top;
        cout << "Stack (top to bottom): ";
        while (current != nullptr) {
            cout << current->data << " ";
            current = current->next;
        }
        cout << endl;
    }
};

// Template Queue
template <typename T>
class Queue {
private:
    struct Node {
        T data;
        shared_ptr<Node> next;
        Node(T val) : data(val), next(nullptr) {}
    };
    
    shared_ptr<Node> front;
    shared_ptr<Node> rear;
    int count;
    
public:
    Queue() : front(nullptr), rear(nullptr), count(0) {}
    
    void enqueue(T value) {
        auto newNode = make_shared<Node>(value);
        
        if (isEmpty()) {
            front = rear = newNode;
        } else {
            rear->next = newNode;
            rear = newNode;
        }
        count++;
    }
    
    T dequeue() {
        if (isEmpty()) {
            throw runtime_error("Queue is empty!");
        }
        
        T value = front->data;
        front = front->next;
        
        if (front == nullptr) {
            rear = nullptr;
        }
        
        count--;
        return value;
    }
    
    T getFront() const {
        if (isEmpty()) {
            throw runtime_error("Queue is empty!");
        }
        return front->data;
    }
    
    bool isEmpty() const {
        return front == nullptr;
    }
    
    int size() const {
        return count;
    }
    
    void display() const {
        auto current = front;
        cout << "Queue (front to rear): ";
        while (current != nullptr) {
            cout << current->data << " ";
            current = current->next;
        }
        cout << endl;
    }
};

// Template Binary Search Tree
template <typename T>
class BST {
private:
    struct Node {
        T data;
        shared_ptr<Node> left, right;
        Node(T val) : data(val), left(nullptr), right(nullptr) {}
    };
    
    shared_ptr<Node> root;
    
    void inorderHelper(shared_ptr<Node> node) const {
        if (node != nullptr) {
            inorderHelper(node->left);
            cout << node->data << " ";
            inorderHelper(node->right);
        }
    }
    
public:
    BST() : root(nullptr) {}
    
    void insert(T value) {
        auto newNode = make_shared<Node>(value);
        
        if (root == nullptr) {
            root = newNode;
            return;
        }
        
        auto current = root;
        while (true) {
            if (value < current->data) {
                if (current->left == nullptr) {
                    current->left = newNode;
                    break;
                }
                current = current->left;
            } else {
                if (current->right == nullptr) {
                    current->right = newNode;
                    break;
                }
                current = current->right;
            }
        }
    }
    
    bool search(T value) const {
        auto current = root;
        while (current != nullptr) {
            if (value == current->data) {
                return true;
            } else if (value < current->data) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        return false;
    }
    
    void inorder() const {
        cout << "Inorder traversal: ";
        inorderHelper(root);
        cout << endl;
    }
};

int main() {
    // Test Stack
    cout << "=== Testing Stack ===" << endl;
    Stack<int> intStack;
    intStack.push(10);
    intStack.push(20);
    intStack.push(30);
    intStack.display();
    cout << "Popped: " << intStack.pop() << endl;
    intStack.display();
    
    Stack<string> stringStack;
    stringStack.push("Hello");
    stringStack.push("World");
    stringStack.display();
    
    // Test Queue
    cout << "\n=== Testing Queue ===" << endl;
    Queue<int> intQueue;
    intQueue.enqueue(10);
    intQueue.enqueue(20);
    intQueue.enqueue(30);
    intQueue.display();
    cout << "Dequeued: " << intQueue.dequeue() << endl;
    intQueue.display();
    
    // Test BST
    cout << "\n=== Testing Binary Search Tree ===" << endl;
    BST<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(20);
    bst.insert(40);
    bst.insert(60);
    bst.insert(80);
    
    bst.inorder();
    
    cout << "Search 40: " << (bst.search(40) ? "Found" : "Not found") << endl;
    cout << "Search 100: " << (bst.search(100) ? "Found" : "Not found") << endl;
    
    return 0;
}
