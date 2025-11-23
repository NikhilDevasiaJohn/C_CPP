// Basic Calculator - Demonstrates functions and control flow
#include <stdio.h>

float add(float a, float b) { return a + b; }
float subtract(float a, float b) { return a - b; }
float multiply(float a, float b) { return a * b; }
float divide(float a, float b) { 
    if (b != 0) return a / b;
    printf("Error: Division by zero!\n");
    return 0;
}

int main() {
    float num1, num2, result;
    char operator;
    char choice;
    
    do {
        printf("\n=== Simple Calculator ===\n");
        printf("Enter first number: ");
        scanf("%f", &num1);
        
        printf("Enter operator (+, -, *, /): ");
        scanf(" %c", &operator);
        
        printf("Enter second number: ");
        scanf("%f", &num2);
        
        switch (operator) {
            case '+':
                result = add(num1, num2);
                printf("%.2f + %.2f = %.2f\n", num1, num2, result);
                break;
            case '-':
                result = subtract(num1, num2);
                printf("%.2f - %.2f = %.2f\n", num1, num2, result);
                break;
            case '*':
                result = multiply(num1, num2);
                printf("%.2f * %.2f = %.2f\n", num1, num2, result);
                break;
            case '/':
                result = divide(num1, num2);
                if (num2 != 0) {
                    printf("%.2f / %.2f = %.2f\n", num1, num2, result);
                }
                break;
            default:
                printf("Invalid operator!\n");
        }
        
        printf("\nContinue? (y/n): ");
        scanf(" %c", &choice);
        
    } while (choice == 'y' || choice == 'Y');
    
    printf("Thank you for using the calculator!\n");
    
    return 0;
}
