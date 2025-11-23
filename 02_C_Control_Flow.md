# C Programming Tutorial - Part 2: Control Flow

## Table of Contents
1. [Conditional Statements](#conditional-statements)
2. [Switch Statement](#switch-statement)
3. [Loops](#loops)
4. [Break and Continue](#break-and-continue)
5. [Nested Control Structures](#nested-control-structures)
6. [Practice Exercises](#practice-exercises)

---

## Conditional Statements

### if Statement

```c
#include <stdio.h>

int main() {
    int age = 18;
    
    if (age >= 18) {
        printf("You are an adult.\n");
    }
    
    return 0;
}
```

### if-else Statement

```c
#include <stdio.h>

int main() {
    int number;
    
    printf("Enter a number: ");
    scanf("%d", &number);
    
    if (number % 2 == 0) {
        printf("%d is even.\n", number);
    } else {
        printf("%d is odd.\n", number);
    }
    
    return 0;
}
```

### if-else if-else Ladder

```c
#include <stdio.h>

int main() {
    int score;
    
    printf("Enter your score (0-100): ");
    scanf("%d", &score);
    
    if (score >= 90) {
        printf("Grade: A\n");
    } else if (score >= 80) {
        printf("Grade: B\n");
    } else if (score >= 70) {
        printf("Grade: C\n");
    } else if (score >= 60) {
        printf("Grade: D\n");
    } else {
        printf("Grade: F\n");
    }
    
    return 0;
}
```

### Ternary Operator

```c
// Syntax: condition ? value_if_true : value_if_false

int a = 10, b = 20;
int max = (a > b) ? a : b;

printf("Maximum: %d\n", max);

// Equivalent to:
// if (a > b) max = a;
// else max = b;
```

---

## Switch Statement

The `switch` statement is used for multi-way branching based on the value of an expression.

### Basic Switch

```c
#include <stdio.h>

int main() {
    int day;
    
    printf("Enter day number (1-7): ");
    scanf("%d", &day);
    
    switch (day) {
        case 1:
            printf("Monday\n");
            break;
        case 2:
            printf("Tuesday\n");
            break;
        case 3:
            printf("Wednesday\n");
            break;
        case 4:
            printf("Thursday\n");
            break;
        case 5:
            printf("Friday\n");
            break;
        case 6:
            printf("Saturday\n");
            break;
        case 7:
            printf("Sunday\n");
            break;
        default:
            printf("Invalid day number!\n");
    }
    
    return 0;
}
```

### Switch with Fall-through

```c
#include <stdio.h>

int main() {
    char grade;
    
    printf("Enter your grade (A-F): ");
    scanf(" %c", &grade);
    
    switch (grade) {
        case 'A':
        case 'a':
            printf("Excellent!\n");
            break;
        case 'B':
        case 'b':
            printf("Good job!\n");
            break;
        case 'C':
        case 'c':
            printf("Well done!\n");
            break;
        case 'D':
        case 'd':
            printf("You passed.\n");
            break;
        case 'F':
        case 'f':
            printf("Better luck next time.\n");
            break;
        default:
            printf("Invalid grade!\n");
    }
    
    return 0;
}
```

> [!IMPORTANT]
> Always use `break` in switch cases unless you intentionally want fall-through behavior!

---

## Loops

### for Loop

Best used when you know the number of iterations in advance.

```c
#include <stdio.h>

int main() {
    // Basic for loop
    for (int i = 0; i < 5; i++) {
        printf("Iteration %d\n", i);
    }
    
    // Counting backwards
    for (int i = 10; i >= 1; i--) {
        printf("%d ", i);
    }
    printf("\n");
    
    // Step by 2
    for (int i = 0; i <= 10; i += 2) {
        printf("%d ", i);
    }
    printf("\n");
    
    return 0;
}
```

### while Loop

Best used when the number of iterations is unknown.

```c
#include <stdio.h>

int main() {
    int count = 0;
    
    while (count < 5) {
        printf("Count: %d\n", count);
        count++;
    }
    
    // Example: Input validation
    int number;
    printf("Enter a positive number: ");
    scanf("%d", &number);
    
    while (number <= 0) {
        printf("Invalid! Enter a positive number: ");
        scanf("%d", &number);
    }
    
    printf("You entered: %d\n", number);
    
    return 0;
}
```

### do-while Loop

Executes at least once, then checks the condition.

```c
#include <stdio.h>

int main() {
    int choice;
    
    do {
        printf("\n=== Menu ===\n");
        printf("1. Option 1\n");
        printf("2. Option 2\n");
        printf("3. Option 3\n");
        printf("0. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);
        
        switch (choice) {
            case 1:
                printf("You selected Option 1\n");
                break;
            case 2:
                printf("You selected Option 2\n");
                break;
            case 3:
                printf("You selected Option 3\n");
                break;
            case 0:
                printf("Exiting...\n");
                break;
            default:
                printf("Invalid choice!\n");
        }
    } while (choice != 0);
    
    return 0;
}
```

---

## Break and Continue

### break Statement

Exits the loop immediately.

```c
#include <stdio.h>

int main() {
    // Find first number divisible by 7
    for (int i = 1; i <= 100; i++) {
        if (i % 7 == 0) {
            printf("First number divisible by 7: %d\n", i);
            break;  // Exit loop
        }
    }
    
    // Search in array
    int numbers[] = {3, 7, 2, 9, 5, 1};
    int target = 9;
    int found = 0;
    
    for (int i = 0; i < 6; i++) {
        if (numbers[i] == target) {
            printf("Found %d at index %d\n", target, i);
            found = 1;
            break;
        }
    }
    
    if (!found) {
        printf("%d not found\n", target);
    }
    
    return 0;
}
```

### continue Statement

Skips the current iteration and continues with the next.

```c
#include <stdio.h>

int main() {
    // Print odd numbers only
    for (int i = 1; i <= 10; i++) {
        if (i % 2 == 0) {
            continue;  // Skip even numbers
        }
        printf("%d ", i);
    }
    printf("\n");
    
    // Skip multiples of 3
    for (int i = 1; i <= 20; i++) {
        if (i % 3 == 0) {
            continue;
        }
        printf("%d ", i);
    }
    printf("\n");
    
    return 0;
}
```

---

## Nested Control Structures

### Nested if Statements

```c
#include <stdio.h>

int main() {
    int age;
    char hasLicense;
    
    printf("Enter your age: ");
    scanf("%d", &age);
    
    if (age >= 18) {
        printf("Do you have a license? (y/n): ");
        scanf(" %c", &hasLicense);
        
        if (hasLicense == 'y' || hasLicense == 'Y') {
            printf("You can drive!\n");
        } else {
            printf("You need a license to drive.\n");
        }
    } else {
        printf("You are too young to drive.\n");
    }
    
    return 0;
}
```

### Nested Loops

```c
#include <stdio.h>

int main() {
    // Multiplication table
    printf("Multiplication Table (1-10):\n\n");
    
    for (int i = 1; i <= 10; i++) {
        for (int j = 1; j <= 10; j++) {
            printf("%4d", i * j);
        }
        printf("\n");
    }
    
    return 0;
}
```

### Pattern Printing

```c
#include <stdio.h>

int main() {
    int n = 5;
    
    // Right triangle
    printf("Right Triangle:\n");
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= i; j++) {
            printf("* ");
        }
        printf("\n");
    }
    
    // Pyramid
    printf("\nPyramid:\n");
    for (int i = 1; i <= n; i++) {
        // Print spaces
        for (int j = 1; j <= n - i; j++) {
            printf(" ");
        }
        // Print stars
        for (int j = 1; j <= 2 * i - 1; j++) {
            printf("*");
        }
        printf("\n");
    }
    
    return 0;
}
```

---

## 📝 Practice Questions (Fill in as you learn)

### Question 1: Conditional Statements
**Q:** What is the difference between `if-else` and `switch` statements?  
**A:** _________________________________

**Q:** Can you use floating-point numbers in a `switch` statement?  
**A:** _________________________________

**Q:** What does the ternary operator `? :` do?  
**A:** _________________________________

### Question 2: Loops
**Q:** What is the difference between `while` and `do-while` loops?  
**A:** _________________________________

**Q:** When should you use a `for` loop vs a `while` loop?  
**A:** _________________________________

**Q:** What happens if you forget the increment in a `for` loop?  
**A:** _________________________________

### Question 3: Break and Continue
**Q:** What does `break` do in a loop?  
**A:** _________________________________

**Q:** What does `continue` do in a loop?  
**A:** _________________________________

**Q:** Can you use `break` outside of a loop or switch?  
**A:** _________________________________

### Question 4: Nested Structures
**Q:** What is the maximum nesting level for loops in C?  
**A:** _________________________________

**Q:** How do you exit from a nested loop?  
**A:** _________________________________

### Question 5: Performance
**Q:** Which is more efficient: `for` loop or `while` loop?  
**A:** _________________________________

**Q:** Why should you minimize branching in performance-critical code?  
**A:** _________________________________

---

## Practice Exercises

### Exercise 1: Prime Number Checker
Write a program to check if a number is prime.

```c
#include <stdio.h>

int main() {
    int num, isPrime = 1;
    
    printf("Enter a number: ");
    scanf("%d", &num);
    
    // Your code here
    
    return 0;
}
```

### Exercise 2: Factorial Calculator
Calculate the factorial of a number using a loop.
- 5! = 5 × 4 × 3 × 2 × 1 = 120

### Exercise 3: Fibonacci Series
Print the first n numbers of the Fibonacci series.
- 0, 1, 1, 2, 3, 5, 8, 13, 21, ...

### Exercise 4: Number Guessing Game
Create a game where the computer picks a random number (1-100) and the user has to guess it.
- Give hints: "Too high" or "Too low"
- Count the number of attempts

### Exercise 5: Pattern Programs
Create these patterns:

```
Pattern 1:        Pattern 2:        Pattern 3:
*                 * * * * *         1
* *               * * * *           1 2
* * *             * * *             1 2 3
* * * *           * *               1 2 3 4
* * * * *         *                 1 2 3 4 5
```

### Exercise 6: Armstrong Number
Check if a number is an Armstrong number.
- An Armstrong number is equal to the sum of cubes of its digits
- Example: 153 = 1³ + 5³ + 3³ = 1 + 125 + 27 = 153

### Exercise 7: Menu-Driven Calculator
Create a calculator with a menu:
1. Addition
2. Subtraction
3. Multiplication
4. Division
0. Exit

Use a do-while loop and switch statement.

---

## Common Patterns and Idioms

### Infinite Loop

```c
// Using while
while (1) {
    // Code
    if (condition) break;
}

// Using for
for (;;) {
    // Code
    if (condition) break;
}
```

### Loop with Multiple Conditions

```c
int i = 0, sum = 0;
while (i < 10 && sum < 50) {
    sum += i;
    i++;
}
```

### Comma Operator in for Loop

```c
for (int i = 0, j = 10; i < j; i++, j--) {
    printf("i = %d, j = %d\n", i, j);
}
```

---

## Performance Tips for CUDA Preparation

> [!TIP]
> **Loop Optimization Concepts** (Important for CUDA):
> 
> 1. **Loop Unrolling**: Reduces loop overhead
> 2. **Minimizing Branching**: Branches can be expensive on GPUs
> 3. **Data Locality**: Access data sequentially when possible
> 4. **Avoiding Nested Loops**: Consider if operations can be parallelized

```c
// Example: Minimize branching
// Instead of:
for (int i = 0; i < n; i++) {
    if (i % 2 == 0) {
        evenSum += arr[i];
    } else {
        oddSum += arr[i];
    }
}

// Consider:
for (int i = 0; i < n; i += 2) {
    evenSum += arr[i];
}
for (int i = 1; i < n; i += 2) {
    oddSum += arr[i];
}
```

---

## Key Takeaways

1. **if-else**: Use for binary or multi-way decisions
2. **switch**: Best for multiple discrete values
3. **for loop**: When iteration count is known
4. **while loop**: When iteration count is unknown
5. **do-while**: When you need at least one execution
6. **break**: Exit loop immediately
7. **continue**: Skip current iteration
8. **Nesting**: Can combine any control structures

---

## Next Steps

In the next tutorial, you'll learn about:
- Functions and modular programming
- Function parameters and return values
- Recursion
- Scope and lifetime of variables

> [!NOTE]
> Understanding control flow is crucial for CUDA because you'll be writing kernel functions that execute in parallel across thousands of threads!
