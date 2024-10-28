from random import random
while True:
    inside=0
    n=int(input('Enter value:'))
    if n==0:break
    for i in range(n):
        x=random()
        y=random()
        if x**2+y**2<1:
            inside+=1
    print(inside*4/n)
