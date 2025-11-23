// Bank Account System - Demonstrates C++ OOP
#include <iostream>
#include <string>
#include <vector>
using namespace std;

class BankAccount {
private:
    static int nextAccountNumber;
    int accountNumber;
    string ownerName;
    double balance;
    
public:
    BankAccount(string name, double initialBalance) {
        accountNumber = nextAccountNumber++;
        ownerName = name;
        balance = initialBalance;
    }
    
    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            cout << "Deposited $" << amount << " successfully." << endl;
        } else {
            cout << "Invalid deposit amount!" << endl;
        }
    }
    
    void withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            cout << "Withdrawn $" << amount << " successfully." << endl;
        } else if (amount > balance) {
            cout << "Insufficient funds!" << endl;
        } else {
            cout << "Invalid withdrawal amount!" << endl;
        }
    }
    
    void transfer(BankAccount &recipient, double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            recipient.balance += amount;
            cout << "Transferred $" << amount << " to account " 
                 << recipient.accountNumber << " successfully." << endl;
        } else {
            cout << "Transfer failed!" << endl;
        }
    }
    
    void displayInfo() const {
        cout << "\n=== Account Information ===" << endl;
        cout << "Account Number: " << accountNumber << endl;
        cout << "Owner: " << ownerName << endl;
        cout << "Balance: $" << balance << endl;
    }
    
    int getAccountNumber() const { return accountNumber; }
    double getBalance() const { return balance; }
};

int BankAccount::nextAccountNumber = 1001;

class Bank {
private:
    vector<BankAccount> accounts;
    
public:
    void createAccount() {
        string name;
        double initialBalance;
        
        cout << "\nEnter account owner name: ";
        cin.ignore();
        getline(cin, name);
        
        cout << "Enter initial balance: $";
        cin >> initialBalance;
        
        accounts.push_back(BankAccount(name, initialBalance));
        cout << "Account created successfully!" << endl;
        accounts.back().displayInfo();
    }
    
    BankAccount* findAccount(int accountNumber) {
        for (auto &account : accounts) {
            if (account.getAccountNumber() == accountNumber) {
                return &account;
            }
        }
        return nullptr;
    }
    
    void performDeposit() {
        int accountNumber;
        double amount;
        
        cout << "\nEnter account number: ";
        cin >> accountNumber;
        
        BankAccount *account = findAccount(accountNumber);
        if (account) {
            cout << "Enter amount to deposit: $";
            cin >> amount;
            account->deposit(amount);
        } else {
            cout << "Account not found!" << endl;
        }
    }
    
    void performWithdrawal() {
        int accountNumber;
        double amount;
        
        cout << "\nEnter account number: ";
        cin >> accountNumber;
        
        BankAccount *account = findAccount(accountNumber);
        if (account) {
            cout << "Enter amount to withdraw: $";
            cin >> amount;
            account->withdraw(amount);
        } else {
            cout << "Account not found!" << endl;
        }
    }
    
    void performTransfer() {
        int fromAccount, toAccount;
        double amount;
        
        cout << "\nEnter source account number: ";
        cin >> fromAccount;
        cout << "Enter destination account number: ";
        cin >> toAccount;
        
        BankAccount *from = findAccount(fromAccount);
        BankAccount *to = findAccount(toAccount);
        
        if (from && to) {
            cout << "Enter amount to transfer: $";
            cin >> amount;
            from->transfer(*to, amount);
        } else {
            cout << "One or both accounts not found!" << endl;
        }
    }
    
    void displayAllAccounts() {
        if (accounts.empty()) {
            cout << "\nNo accounts available." << endl;
            return;
        }
        
        cout << "\n=== All Accounts ===" << endl;
        for (const auto &account : accounts) {
            account.displayInfo();
        }
    }
};

int main() {
    Bank bank;
    int choice;
    
    do {
        cout << "\n=== Bank Management System ===" << endl;
        cout << "1. Create Account" << endl;
        cout << "2. Deposit" << endl;
        cout << "3. Withdraw" << endl;
        cout << "4. Transfer" << endl;
        cout << "5. Display All Accounts" << endl;
        cout << "0. Exit" << endl;
        cout << "Enter your choice: ";
        cin >> choice;
        
        switch (choice) {
            case 1:
                bank.createAccount();
                break;
            case 2:
                bank.performDeposit();
                break;
            case 3:
                bank.performWithdrawal();
                break;
            case 4:
                bank.performTransfer();
                break;
            case 5:
                bank.displayAllAccounts();
                break;
            case 0:
                cout << "Thank you for using our bank!" << endl;
                break;
            default:
                cout << "Invalid choice!" << endl;
        }
    } while (choice != 0);
    
    return 0;
}
