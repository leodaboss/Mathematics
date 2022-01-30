import math


def normal_remainder(x, y):
    return x % y


def hcf(x, y, remainder):
    if x < y:
        return hcf_helper(y, x, remainder)
    return hcf_helper(x, y, remainder)


def hcf_helper(x, y, remainder):
    if y == 0:
        return x
    z = remainder(x, y)

    # extra stuff if we want to show the computation
    # d=math.floor(x/y)
    # print(x,"=",d,"*",y,"+",z)

    return hcf(y, z, remainder)


def get_int_helper(string, is_limit, limit=0):
    response = ''
    while True:
        try:
            response = int(input(string))
            if not is_limit or response >= limit:
                return response
            else:
                print('Out of range, try again')
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")
    return None


# makes sure we get the right kind of input, ie a number
def get_int(string):
    return get_int_helper(string, False)


# makes sure we get the right kind of input, ie a number>=limit
def get_int_limit(string, limit):
    return get_int_helper(string, True, limit)


# this is to test algorithm
def main():
    limit = 0
    x1 = get_int_limit('First number: ', limit)
    x2 = get_int_limit('Second number: ', limit)
    print(hcf(x1, x2, normal_remainder))


main()
