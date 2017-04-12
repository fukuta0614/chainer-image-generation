
N = 10
for i in range(N):
    if i == 5:
        print('asdf')
        break
else:
    print('else')


N = 10
for i in range(N):
    if i == 5:
        print('asdf')
        break
else:
    print('else')


N = 10
for i in range(N):
    if i == 5:
        print('asdf')
else:
    print('else')

for i in range(10):
    for j in range(10):
        if (i,j) == (10,10):
            break
    else:
        continue
    break
