// Student Management System - Demonstrates structures and file I/O
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_STUDENTS 100

typedef struct {
    int id;
    char name[50];
    float gpa;
} Student;

Student students[MAX_STUDENTS];
int studentCount = 0;

void addStudent() {
    if (studentCount >= MAX_STUDENTS) {
        printf("Maximum students reached!\n");
        return;
    }
    
    Student s;
    printf("\nEnter student ID: ");
    scanf("%d", &s.id);
    
    printf("Enter student name: ");
    scanf(" %[^\n]", s.name);
    
    printf("Enter GPA: ");
    scanf("%f", &s.gpa);
    
    students[studentCount++] = s;
    printf("Student added successfully!\n");
}

void displayAllStudents() {
    if (studentCount == 0) {
        printf("\nNo students to display.\n");
        return;
    }
    
    printf("\n%-10s %-30s %-10s\n", "ID", "Name", "GPA");
    printf("---------------------------------------------------\n");
    for (int i = 0; i < studentCount; i++) {
        printf("%-10d %-30s %-10.2f\n", 
               students[i].id, students[i].name, students[i].gpa);
    }
}

void searchStudent() {
    int id;
    printf("\nEnter student ID to search: ");
    scanf("%d", &id);
    
    for (int i = 0; i < studentCount; i++) {
        if (students[i].id == id) {
            printf("\nStudent found:\n");
            printf("ID: %d\n", students[i].id);
            printf("Name: %s\n", students[i].name);
            printf("GPA: %.2f\n", students[i].gpa);
            return;
        }
    }
    
    printf("Student with ID %d not found.\n", id);
}

void findTopStudent() {
    if (studentCount == 0) {
        printf("\nNo students available.\n");
        return;
    }
    
    int topIndex = 0;
    for (int i = 1; i < studentCount; i++) {
        if (students[i].gpa > students[topIndex].gpa) {
            topIndex = i;
        }
    }
    
    printf("\nTop Student:\n");
    printf("ID: %d\n", students[topIndex].id);
    printf("Name: %s\n", students[topIndex].name);
    printf("GPA: %.2f\n", students[topIndex].gpa);
}

void calculateAverageGPA() {
    if (studentCount == 0) {
        printf("\nNo students available.\n");
        return;
    }
    
    float sum = 0;
    for (int i = 0; i < studentCount; i++) {
        sum += students[i].gpa;
    }
    
    printf("\nAverage GPA: %.2f\n", sum / studentCount);
}

int main() {
    int choice;
    
    do {
        printf("\n=== Student Management System ===\n");
        printf("1. Add Student\n");
        printf("2. Display All Students\n");
        printf("3. Search Student\n");
        printf("4. Find Top Student\n");
        printf("5. Calculate Average GPA\n");
        printf("0. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);
        
        switch (choice) {
            case 1:
                addStudent();
                break;
            case 2:
                displayAllStudents();
                break;
            case 3:
                searchStudent();
                break;
            case 4:
                findTopStudent();
                break;
            case 5:
                calculateAverageGPA();
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
