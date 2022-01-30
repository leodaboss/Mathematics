# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:26:55 2022

@author: leoda
"""
precedence = {
    '+': 1,
    '-': 1,
    '*': 2,
    '/': 2,
    '^': 3,
    None: 0
}
symbols = ['+', '-', '*', '/', '^']
all_allowed = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', ')', '(']
all_allowed.extend(symbols)


# evaluates a simple expression
def evaluate_simple(x, y, symbol):
    value = None
    if symbol == '+':
        value = x + y
    elif symbol == '-':
        value = x - y
    elif symbol == '*':
        value = x * y
    elif symbol == '/':
        value = x / y
    elif symbol == '^':
        value = x ** y
    return value


# evaluates the equation in polish notation by recursion, equation is an array
def evaluate_in_polish(equation):
    # can't have even length or length 0 as there is one more number than symbol
    if equation is None or len(equation) % 2 == 0:
        return None

    # works by replacing elements by result until there is one left, the answer
    while len(equation) > 1:
        # find the first symbol element
        counter = 2
        while equation[counter] not in symbols:
            counter += 1

        # once found first symbol, evaluate its expression
        result = evaluate_simple(equation[counter - 2], equation[counter - 1], equation[counter])

        # replaces operands and operator with answer
        equation.pop(counter - 2)
        equation.pop(counter - 2)
        equation[counter - 2] = result
    return equation[0]


# allows either the use of integers or floating point numbers
def to_polish(equation, type_number):
    # initialising data, in_polish is list in RPN, lower is used to extract numbers
    # stack_symbols is all symbols of lower precedence, i is used to iterate
    in_polish = []
    lower = 0
    stack_symbols = []
    i = 0
    end_bracket = False
    while i < len(equation) - 1:
        if equation[i] == '(':
            # lower is first place after the bracket
            i += 1
            lower = i

            # finds opening bracket corresponding to opening bracket,end just after it
            brackets = 0
            while brackets < 1:
                # hasn't found a second bracket, returns nothing
                if i >= len(equation):
                    print('Index out of range, wasnt correct bracketing!')
                    return None
                if equation[i] == '(': brackets -= 1
                if equation[i] == ')': brackets += 1
                i += 1

            # Show what was in the brackets
            # print('bracket encountered: '+equation[lower:i-1])

            # adds the stuff in the brackets in RPN in right order and reinitialises lower
            in_polish.extend(to_polish(equation[lower:i - 1], type_number))
            lower = i

            # last encountered was bracketed
            end_bracket = True
        elif equation[i] in symbols:
            # find number and makes sure it is correct
            # add number to the list if last wasn't bracket
            if not end_bracket:
                try:
                    number = type_number(equation[lower:i])
                    in_polish.append(number)
                except ValueError:
                    print('Not a correct number between symbols before end!')
                    return None
            # reinitialise end_bracket and calculate precedence of symbol
            end_bracket = False
            precede = precedence[equation[i]]

            # for all symbols which are already if they have same precedence or higher
            # they need to be added to the list
            for j in range(len(stack_symbols)):
                if precedence[stack_symbols[len(stack_symbols) - j - 1]] >= precede:
                    in_polish.append(stack_symbols.pop())
                else:
                    break

            # add the symbol next at the top of the stack of symbols so popped first
            stack_symbols.append(equation[i])

            # adjust i and lower
            i += 1
            lower = i
        else:
            i += 1
        # print(in_polish)

    # if last symbol wasn't a bracket add the number
    if not end_bracket:
        try:
            number = type_number(equation[lower:])
            in_polish.append(number)
        except:
            print('Not a correct number between symbols in end!')
            return None

    # adds the rest of the symbols which were on the stack fo lower precedence
    for j in range(len(stack_symbols)):
        in_polish.append(stack_symbols.pop())

    return in_polish


def evaluate(equation, type_number):
    for j in equation:
        if j not in all_allowed:
            print('Symbols used not allowed')
            return None
    return evaluate_in_polish(to_polish(equation, type_number))


def evaluate_int(equation):
    return evaluate(equation, int)


def evaluate_float(equation):
    return evaluate(equation, float)


def main():
    equation2 = '3*(9-6)^2'
    # print(evaluate_int(equation2))
    answer = None
    in_polish = None
    while answer == None:
        equation2 = input('What is your equation: ')
        in_polish = to_polish(equation2, int)
        print(in_polish)
        answer = evaluate_in_polish(in_polish)
    print('Evaluated ' + equation2 + ' gives: ' + str(answer))


main()
