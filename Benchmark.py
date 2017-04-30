import time

start = time.time()
cont = 0
for i in range(0,1000000):
    cont += 1
end = time.time()
print(end - start)