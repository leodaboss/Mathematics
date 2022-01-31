class Matrix:
    bad = 'Improper'
    bad_dimension = 'Incorrect dimensions'

    def __init__(self, array):

        self.__coefficients = array

    @staticmethod
    def constant_matrix(n, m, constant):
        if n <= 0 or m <= 0:
            print(Matrix.bad)
            return
        return Matrix([[constant for i in range(m)] for j in range(n)])

    @staticmethod
    def zeros(n, m):
        return Matrix.constant_matrix(n, m, 0)

    @staticmethod
    def ones(n, m):
        return Matrix.constant_matrix(n, m, 1)

    @property
    def coefficients(self):
        return self.__coefficients

    @coefficients.setter
    def coefficients(self, value):
        self.__coefficients = value

    @property
    def rows(self):
        return len(self.coefficients)

    @property
    def columns(self):
        return len(self.coefficients[0])

    def set_coefficient(self, value, row, column):
        if self.check_coefficient(row, column):
            self.__coefficients[row][column] = value

    def change_coefficient(self, value, row, column):
        if self.check_coefficient(row, column):
            self.__coefficients[row][column] += value

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
        self.coefficients = new

    def multiply_constant(self, constant):
        matrix = self.coefficients
        new = [[constant * matrix[i][j] for j in range(self.columns)] for i in range(self.rows)]
        return Matrix(new)

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

    @property
    def transpose(self):
        matrix = self.coefficients
        new = [[matrix[i][j] for i in range(self.rows)] for j in range(self.columns)]
        return Matrix(new)

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

    @property
    def determinant(self):
        if self.columns != self.rows:
            print(Matrix.bad_dimension)
            return
        return Matrix.determinant_helper(self.coefficients)

    @staticmethod
    def determinant_helper(matrix):
        n = len(matrix)
        if n == 1:
            return matrix[0][0]
        reduced = Matrix.delete_row_list(matrix, 0)
        return sum([matrix[0][j] * (-1) ** j * Matrix.determinant_helper(Matrix.delete_column_list(reduced, j)) for j in
                    range(n)])
    def power(self, n):
        if n < 0 or self.columns!=self.rows:
            return
        return self.power_helper(n)

    def power_helper(self, n):
        if n == 0:
            return Matrix(Matrix.ones(n,n))
        if n == 1:
            return self.__copy__()
        return self * self.power_helper(n - 1)
    def __copy__(self):
        return Matrix(self.coefficients.copy())


def main():
    test = [[1, 2, 3], [3, 4, 5], [9, 6, 0]]
    matrix1 = Matrix.ones(2, 9)
    matrix2 = Matrix.ones(2, 9)
    matrix3 = Matrix(test)
    print(matrix1+matrix2)
    print(matrix3)
    print(matrix3.determinant)
    print(matrix1)
    print(matrix1.transpose)
    print(Matrix.delete_row(matrix1, 1))
    print(Matrix.delete_column(matrix1, 1))
    print(matrix1)
    print(matrix1.columns)
    # print(Matrix.multiply_matrices(matrix1, matrix1.transpose))
    # matrix1.add(matrix2)
    # print(matrix1)
    print(matrix3.power(3))
    print(matrix3)


main()
