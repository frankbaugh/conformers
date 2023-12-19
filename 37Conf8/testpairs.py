pairs = []
    ## Given length of 
for i in range(3):
    for j in range(i+1, 3):
        pairs.append((i, j))
        pairs.append((j, i))

print(pairs)

