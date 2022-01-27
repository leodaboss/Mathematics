pentagonal=[1]
current=1
for i in range(10000):
    #at the end
    current+=3*i+4
    pentagonal.append(current)
print('cool')
for i in range(len(pentagonal)):
    a=pentagonal[i]
    for j in range(i):
        b=pentagonal[j]
        if a-b in pentagonal and a+b in pentagonal:
            print(a,b,a-b)
print('yay')
