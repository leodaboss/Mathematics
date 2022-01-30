# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 11:11:54 2022

@author: leoda
"""


class Polynomials:
    def __init__(self, coeffs=[]):
        self.__coefficients = coeffs

    @property
    def coefficients(self):
        return self.__coefficients

    @coefficients.setter
    def coefficients(self, value):
        self.__coefficients = value
        self.__check_zeros()

    @property
    def degree(self):
        return len(self.coefficients)

    def __str__(self):
        string = 'f(x) = '
        for i in range(self.degree):
            value = self.coefficients[i]
            if value != 0:
                string += str(value)
                if i > 0:
                    string += 'x'
                if i > 1:
                    string += '^' + str(i)
                string += ' + '

        return string[:len(string) - 2]

    def __set_coefficients(self, index, number):
        if index < 0:
            print('not possible')
        if index < self.degree:
            self.coefficients[index] = number
            if number == 0 and index == self.degree - 1:
                self.__check_zeros()
        if index >= self.degree:
            zeros = [0 for i in range(index - self.degree)]
            self.coefficients.extend(zeros)
            self.coefficients.append(number)

    def __change_coefficients(self, index, number):
        current = self.__get_coefficients(self, index)
        self.__set_coefficients(self, index, number + current)

    def __get_coefficients(self, index):
        if index >= self.degree or index < 0:
            print('not possible')
        return self.coefficients[index]

    # Check zeros at start of list of coefficients and changes it accordingly
    def __check_zeros(self):
        if not self.coefficients:
            return
        i = self.degree - 1
        while self.coefficients[i] == 0:
            i -= 1
        if i != self.degree - 1:
            self.coefficients = self.coefficients[:i + 1]

    def add(self, operand):
        for i in range(operand.__degree):
            self.__change_coefficients(i, operand.__get_coefficients(i))

    def subtract(self, operand):
        for i in range(operand.__degree):
            self.__change_coefficients(i, -operand.__get_coefficients(i))

    def multiply_constant(self, constant):
        if constant == 0:
            self.coefficients = []
        self.coefficients = [i * constant for i in self.coefficients]

    def multiply(self, operand):
        new=[0 for i in range(self.degree+operand.degree)]
        for i in range(self.degree):
            for j in range(operand.degree):
                new[i+j]+=self.coefficients[i]*operand.coefficients[j]
        self.coefficients=new

    def evaluate(self, number):
        return sum([self.coefficients[i] * number ** i for i in range(self.degree)])
    def remainder(self):
        return

# Test code
def main():
    coeffs1 = [6, 1, 2, 6, 4]
    coeffs2=[2,3,4]
    pol1 = Polynomials(coeffs1)
    pol2 = Polynomials(coeffs2)
    print(pol1)
    pol1.multiply_constant(2)
    print(pol1)
    print(pol1.evaluate(1))
    print(pol2)
    pol1.multiply(pol2)
    print(pol1)
main()
