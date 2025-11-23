// Matrix Operations - Demonstrates 2D arrays and pointers
#include <stdio.h>
#include <stdlib.h>

void printMatrix(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%4d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void addMatrices(int **a, int **b, int **result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
}

void multiplyMatrices(int **a, int **b, int **result, int r1, int c1, int c2) {
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < c1; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main() {
    int rows = 3, cols = 3;
    
    // Allocate matrices
    int **matrix1 = (int **)malloc(rows * sizeof(int *));
    int **matrix2 = (int **)malloc(rows * sizeof(int *));
    int **sum = (int **)malloc(rows * sizeof(int *));
    int **product = (int **)malloc(rows * sizeof(int *));
    
    for (int i = 0; i < rows; i++) {
        matrix1[i] = (int *)malloc(cols * sizeof(int));
        matrix2[i] = (int *)malloc(cols * sizeof(int));
        sum[i] = (int *)malloc(cols * sizeof(int));
        product[i] = (int *)malloc(cols * sizeof(int));
    }
    
    // Initialize matrices
    printf("Enter elements of Matrix 1 (%dx%d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            scanf("%d", &matrix1[i][j]);
        }
    }
    
    printf("Enter elements of Matrix 2 (%dx%d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            scanf("%d", &matrix2[i][j]);
        }
    }
    
    // Display matrices
    printf("\nMatrix 1:\n");
    printMatrix(matrix1, rows, cols);
    
    printf("\nMatrix 2:\n");
    printMatrix(matrix2, rows, cols);
    
    // Addition
    addMatrices(matrix1, matrix2, sum, rows, cols);
    printf("\nMatrix 1 + Matrix 2:\n");
    printMatrix(sum, rows, cols);
    
    // Multiplication
    multiplyMatrices(matrix1, matrix2, product, rows, cols, cols);
    printf("\nMatrix 1 * Matrix 2:\n");
    printMatrix(product, rows, cols);
    
    // Free memory
    for (int i = 0; i < rows; i++) {
        free(matrix1[i]);
        free(matrix2[i]);
        free(sum[i]);
        free(product[i]);
    }
    free(matrix1);
    free(matrix2);
    free(sum);
    free(product);
    
    return 0;
}
