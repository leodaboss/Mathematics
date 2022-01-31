# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 11:11:54 2022

@author: leoda
"""


class Polynomial:
    def __init__(self, coeffs=[]):
        self.__coefficients = coeffs
        self.check_zeros()

    @property
    def coefficients(self):
        return self.__coefficients

    @coefficients.setter
    def coefficients(self, value):
        self.__coefficients = value
        self.check_zeros()

    @property
    def degree(self):
        return len(self.coefficients)
    @property
    def leading_coefficient(self):
        return self.coefficients[-1]

    @staticmethod
    def get_power_lead(n,lead):
        new=[0 for i in range(n)]
        new.append(lead)
        return Polynomial(new)
    @staticmethod
    def get_power(n):
        return Polynomial.get_power_lead(n,1)
    @staticmethod
    def get_one():
        return Polynomial.get_power(0,1)
    @staticmethod
    def get_x():
        return Polynomial.get_power(1,1)
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
                self.check_zeros()
        if index >= self.degree:
            zeros = [0 for i in range(index - self.degree)]
            self.coefficients.extend(zeros)
            self.coefficients.append(number)

    def change_coefficient(self, index, number):
        current = self.get_coefficient(index)
        self.__set_coefficients( index, number + current)

    def get_coefficient(self, index):
        if index < 0:
            print('not possible')
        if index>=self.degree:
            return 0
        return self.coefficients[index]

    # Check zeros at start of list of coefficients and changes it accordingly
    def check_zeros(self):
        if not self.coefficients:
            return
        i = self.degree - 1
        while self.coefficients[i] == 0:
            i -= 1
        if i != self.degree - 1:
            self.coefficients = self.coefficients[:i + 1]

    def __add__(self, operand):
        n=max(self.degree,operand.degree)
        coeffs=[self.get_coefficient(i)+operand.get_coefficient(i) for i in range(n)]
        return Polynomial(coeffs)

    def __sub__(self, operand):
        n = max(self.degree, operand.degree)
        coeffs = [self.get_coefficient(i) - operand.get_coefficient(i) for i in range(n)]
        return Polynomial(coeffs)

    def multiply_constant(self, constant):
        return Polynomial([i * constant for i in self.coefficients])

    def __mul__(self, operand):
        new = [0 for i in range(self.degree + operand.degree-1)]
        for i in range(self.degree):
            for j in range(operand.degree):
                new[i + j] += self.get_coefficient(i) * operand.get_coefficient(j)
        return Polynomial(new)

    def evaluate(self, number):
        return sum([self.coefficients[i] * number ** i for i in range(self.degree)])

    def __mod__(self,operand):
        if operand.degree==0:
            print('Impossible')
            return
        d = self.degree - operand.degree
        if d<0:
            return Polynomial(self.coefficients[:])
        subtract=operand*Polynomial.get_power_lead(d,self.leading_coefficient/operand.leading_coefficient)
        return (self-subtract)%operand
def main():
    coeffs1 = [6, 1, 2, 6, 4]
    coeffs2 = [2, 3, 4]
    pol1 = Polynomial(coeffs1)
    pol2 = Polynomial(coeffs2)
    print(pol2.leading_coefficient)
    print(pol1)
    print(pol1.multiply_constant(2))
    print(pol1.evaluate(1))
    print(pol2)
    pol3=pol1*pol2
    print('Multiplied gives: ')
    print(pol3)
    print('remainder:')
    print(pol1%Polynomial.get_power(2))
    print(pol3.leading_coefficient)
    print(pol3.coefficients)


main()
