import random

from Polynomials import Polynomial


class Matrix:
    bad = 'Improper'
    bad_dimension = 'Incorrect dimensions'

    def __init__(self, array):
        self.__coefficients = array

    #a bunch of useful properties
    @property
    def coefficients(self):
        return self.__coefficients

    @coefficients.setter
    def coefficients(self, value):
        self.__coefficients = value

    @property
    def transpose(self):
        matrix = self.coefficients
        new = [[matrix[i][j] for i in range(self.rows)] for j in range(self.columns)]
        return Matrix(new)

    @property
    def rows(self):
        return len(self.coefficients)

    @property
    def columns(self):
        return len(self.coefficients[0])

    @property
    def is_square(self):
        return self.columns == self.rows

    #useful functions to simplify computation
    def set_coefficient(self, row, column, value):
        if self.check_coefficient(row, column):
            self.__coefficients[row][column] = value

    def change_coefficient(self, value, row, column):
        if self.check_coefficient(row, column):
            self.set_coefficient(row, column, self.get_coefficient(row, column) + value)

    def get_coefficient(self, row, column):
        if self.check_coefficient(row, column):
            return self.coefficients[row][column]

    def check_dimensions(self, operand):
        return self.columns == operand.columns and self.rows == operand.rows

    def check_coefficient(self, row, column):
        return 0 <= row < self.rows and 0 <= column < self.columns

    def multiply_constant(self, constant):
        matrix = self.coefficients
        new = [[constant * matrix[i][j] for j in range(self.columns)] for i in range(self.rows)]
        return Matrix(new)

    #generate generic matrices
    @staticmethod
    def constant(n, m, constant):
        if n <= 0 or m <= 0:
            print(Matrix.bad)
            return
        return Matrix([[constant for i in range(m)] for j in range(n)])

    @staticmethod
    def zeros(n, m):
        return Matrix.constant(n, m, 0)

    @staticmethod
    def ones(n, m):
        return Matrix.constant(n, m, 1)

    #generate random matrices with coefficients in a uniform distribution
    @staticmethod
    def generate_random(n, m, big):
        return Matrix([[random.random() * big for j in range(m)] for i in range(n)])

    @staticmethod
    def generate_random_uniform(n, m):
        return Matrix.generate_random(n, m, 1)

    #overrides the basic operations for matrices
    def __add__(self, operand):
        if not self.check_dimensions(operand):
            print(Matrix.bad_dimension)
            return
        matrix1 = self.coefficients
        matrix2 = operand.coefficients
        new = [[matrix1[i][j] + matrix2[i][j] for j in range(self.columns)] for i in range(self.rows)]
        return Matrix(new)

    def __sub__(self, operand):
        return self + Matrix.multiply_constant(operand, -1)

    def __mul__(self, operand):
        if self.columns != operand.rows:
            print(Matrix.bad_dimension)
            return
        common = self.columns
        matrix1 = self.coefficients
        matrix2 = operand.coefficients
        n = self.rows
        m = operand.columns
        new = [[sum([matrix1[i][k] * matrix2[k][j] for k in range(common)]) for j in range(m)] for i in range(n)]
        return Matrix(new)

    def __str__(self):
        string = ''
        for j in self.coefficients:
            string += '['
            for i in j:
                string += ' ' + str(i) + ' '
            string += ']\n'
        return string

    def __copy__(self):
        return Matrix(self.coefficients.copy())

    #useful methods for computation
    @staticmethod
    def delete_column_list(matrix, n):
        if n < 0 or n >= len(matrix[0]):
            print(Matrix.bad_dimension)
            return
        matrix1 = [j.copy() for j in matrix]
        for j in matrix1:
            j.pop(n)
        return matrix1

    @staticmethod
    def delete_column(operand, n):
        return Matrix(Matrix.delete_column_list(operand.coefficients, n))

    @staticmethod
    def delete_row_list(matrix, n):
        if n < 0 or n >= len(matrix):
            print(Matrix.bad_dimension)
            return
        matrix1 = matrix.copy()
        matrix1.pop(n)
        return matrix1

    @staticmethod
    def delete_row(operand, n):
        return Matrix(Matrix.delete_row_list(operand.coefficients, n))

#a special class for square matrices with extra functionality
class SquareMatrix(Matrix):

    #sets up array then calculates characteristic poly
    def __init__(self, array):
        super().__init__(array)
        self.__characteristic = SquareMatrix.characteristic_helper(array)

    @property
    def characteristic(self):
        return self.__characteristic

    #overrides columns property since square dimension to make it easier
    @property
    def columns(self):
        return self.rows
    #returns sum of diagonal entries, the trace
    @property
    def trace(self):
        return sum([self.get_coefficient(i, i) for i in range(self.rows)])

    #evaluates determinant as characteristic poly at 0
    @property
    def determinant(self):
        return self.characteristic.evaluate(0)

    #calculates the determinant for a matrix containing poly values
    @staticmethod
    def determinant_helper(matrix):
        n = len(matrix)
        if n == 1:
            return matrix[0][0]
        reduced = Matrix.delete_row_list(matrix, 0)
        coeffs = [matrix[0][j] * Polynomial.constant((-1) ** j) for j in range(n)]
        smaller = [SquareMatrix.determinant_helper(Matrix.delete_column_list(reduced, j)) for j in range(n)]
        initial = Polynomial()
        for j in range(n):
            initial += coeffs[j] * smaller[j]
        return initial

    #takes a matrix and calculates the characteristic poly by turning it into a matrix of polynomials
    @staticmethod
    def characteristic_helper(matrix):
        n = len(matrix)
        polynomials = [[Polynomial.constant(matrix[i][j]) for j in range(n)] for i in range(n)]
        for i in range(n):
            polynomials[i][i] -= Polynomial.get_x()
        return SquareMatrix.determinant_helper(polynomials)

    #creates generic matrix of certain size with constant values
    @staticmethod
    def constant_matrix(n, constant):
        if n <= 0:
            print(Matrix.bad)
            return
        return SquareMatrix([[constant for i in range(n)] for j in range(n)])

    @staticmethod
    def zero(n):
        return SquareMatrix.constant_matrix(n, 0)

    @staticmethod
    def ones(n):
        return SquareMatrix.constant_matrix(n, 1)

    @staticmethod
    def identity(n):
        return SquareMatrix.identity_constant(n, 1)

    @staticmethod
    def identity_constant(n, constant):
        new = SquareMatrix.zero(n)
        for i in range(n):
            new.set_coefficient(n, n, constant)
        return new

    #generates matrices with coefficients in a uniform distribution
    @staticmethod
    def generate_random(n, range):
        return SquareMatrix([[random.random() * range for j in range(n)] for i in range(n)])

    @staticmethod
    def generate_random_uniform(n):
        return Matrix.generate_random(n, n, 1)

    #overrides methods
    def __pow__(self, n):
        if n < 0:
            return
        if n == 0:
            return SquareMatrix.identity(self.rows)
        if n == 1:
            return self.__copy__()
        return self * self ** (n - 1)



def main():
    test = [[-2,2,0], [2,-3,1], [0,2,-2]]
    matrix4=SquareMatrix(test)
    matrix1 = Matrix.ones(2, 9)
    matrix2 = Matrix.ones(2, 9)
    matrix3 = SquareMatrix(test)
    print(matrix4.characteristic)
    print(matrix4.characteristic.find_root(-1,1.5,0.003))
    # print(matrix1 + matrix2)
    #print(matrix3)
    #print(SquareMatrix.generate_random_uniform(4))
    #print(matrix3.characteristic)
    #print(matrix3 ** 2)
    #(matrix3)
    # print(matrix1)
    # print(matrix1.transpose)
    # print(Matrix.delete_row(matrix1, 1))
    # print(Matrix.delete_column(matrix1, 1))
    # print(matrix1)
    # print(matrix1.columns)
    # print(matrix1*matrix1.transpose))
    # print(matrix1)
    # print(matrix3**3)
    # print(matrix3)


main()
