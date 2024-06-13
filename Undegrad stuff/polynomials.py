# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 11:11:54 2022

@author: leoda
"""
import RootFinding


class Polynomial():
    def __init__(self, coeffs=[]):
        self.__coefficients = coeffs
        self.check_zeros()

    # properties and setters
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
        return self.get_coefficient(self.degree - 1)

    @property
    def derivative(self):
        if self.degree <= 1:
            return Polynomial.zero()
        else:
            return Polynomial([(i + 1) * self.get_coefficient(i + 1) for i in range(self.degree - 1)])

    # generates generic polynomials
    @staticmethod
    def get_power_lead(n, lead):
        new = [0 for i in range(n)]
        new.append(lead)
        return Polynomial(new)

    @staticmethod
    def get_power(n):
        return Polynomial.get_power_lead(n, 1)

    @staticmethod
    def constant(constant):
        return Polynomial.get_power_lead(0, constant)

    # polynomials that are regularly used
    @staticmethod
    def identity():
        return Polynomial([1])

    @staticmethod
    def zero():
        return Polynomial([0])

    @staticmethod
    def get_x():
        return Polynomial.get_power(1)

    # manipulation of polynomials
    def set_coefficients(self, index, number):
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
        self.set_coefficients(index, number + current)

    def get_coefficient(self, index):
        if index < 0:
            print('not possible')
        if index >= self.degree:
            return 0
        return self.coefficients[index]

    # Check zeros at start of list of coefficients and changes it accordingly
    def check_zeros(self):
        if not self.coefficients:
            self.__coefficients=[0]
        i = self.degree - 1
        while self.coefficients[i] == 0:
            if i == -1:
                self.__coefficients = [0]
                return
            i -= 1
        if i != self.degree - 1:
            self.coefficients = self.coefficients[:i + 1]

    # returns polynomial evaluated at a value
    def evaluate(self, number):
        return sum([self.get_coefficient(i) * number ** i for i in range(self.degree)])

    #returns evaluator
    @property
    def evaluator(self):
        return lambda x: self.evaluate(x)

    # overrides methods
    def __add__(self, operand):
        n = max(self.degree, operand.degree)
        coeffs = [self.get_coefficient(i) + operand.get_coefficient(i) for i in range(n)]
        return Polynomial(coeffs)

    def __sub__(self, operand):
        n = max(self.degree, operand.degree)
        coeffs = [self.get_coefficient(i) - operand.get_coefficient(i) for i in range(n)]
        return Polynomial(coeffs)

    def __mul__(self, operand):
        new = [0 for i in range(self.degree + operand.degree - 1)]
        for i in range(self.degree):
            for j in range(operand.degree):
                new[i + j] += self.get_coefficient(i) * operand.get_coefficient(j)
        return Polynomial(new)

    def __mod__(self, operand):
        if operand.degree == 0:
            print('Impossible')
            return
        d = self.degree - operand.degree
        if d < 0:
            return Polynomial(self.coefficients[:])
        subtract = operand * Polynomial.get_power_lead(d, self.leading_coefficient / operand.leading_coefficient)
        return (self - subtract) % operand

    def __pow__(self, n):
        if n < 0:
            return
        if n == 0:
            return Polynomial.identity()
        if n == 1:
            return self.__copy__()
        return self * self ** (n - 1)

    def __str__(self):
        string = 'f(x) = '
        for i in range(self.degree):
            value = self.coefficients[i]
            if value != 0:
                if i != 0 and value == -1:
                    string += '-'
                elif i == 0 or value != 1:
                    string += str(value)
                if i > 0:
                    string += 'x'
                if i > 1:
                    string += '^' + str(i)
                string += ' + '

        return string[:len(string) - 2]

    def __copy__(self):
        return Polynomial(self.coefficients.copy())



    # interpolation methods
    @staticmethod
    def interpolation(values, interpolants):
        n = len(values)
        if n != len(interpolants):
            print('Bad dimension')
            return
        new = Polynomial()
        for i in range(n):
            current = Polynomial.constant(interpolants[i])
            for j in range(n):
                if j != i:
                    pol = Polynomial.get_x() - Polynomial.constant(interpolants[j])
                    current *= pol * Polynomial.constant(1 / (values[i] - values[j]))
            new += current
        return new

    @staticmethod
    def hermite_interpolation(values, interpolants, derivatives):
        n = len(values)
        if n != len(interpolants) or n != len(derivatives):
            print('Bad dimension')
            return
        new = Polynomial()
        for i in range(n):
            current = Polynomial.identity()
            for j in range(n):
                if j != i:
                    pol = Polynomial.get_x() - Polynomial.constant(interpolants[j])
                    current *= pol * Polynomial.constant(1 / (values[i] - values[j]))
            square = current * current
            difference = Polynomial.get_x() - Polynomial.constant(values[i])
            H = square * (Polynomial.identity() - Polynomial.constant(
                2 * current.derivative.evaluate(values[i])) * difference)
            K = square * difference
            new += H * Polynomial.constant(interpolants[i]) + K * Polynomial.constant(derivatives[i])
        return new

    def find_root(self,bottom, top,tol):
        return RootFinding.RootFinding.bisection(bottom, top, tol, self.evaluator)


def main():
    coeffs1 = [6, 1, 2, 6, 4]
    coeffs2 = [2, 3, 4]
    values = [0, 1]
    interpolants = [2, 3]
    derivatives = [8, 9]
    print(Polynomial.hermite_interpolation(values, interpolants, derivatives))
    pol1 = Polynomial(coeffs1)
    pol2 = Polynomial(coeffs2)
    print(pol2)
    print(Polynomial.zero()**2)
    print(pol2.derivative)
    print(pol2.leading_coefficient)
    print(pol1)
    print(pol1 * Polynomial.constant(2))
    print(pol1.evaluate(1))
    print(pol2)
    pol3 = pol1 * pol2
    print('Multiplied gives: ')
    print(pol3)
    print('remainder:')
    print(pol1 % Polynomial.get_power(2))
    print(pol3.leading_coefficient)
    print(pol3.coefficients)
    print(pol3 ** 3)
    print(pol3.evaluator)

#main()
