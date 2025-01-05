import os

path = "/dtu/blackhole/05/146725/shards/shards/mp16/"

num_countries = 0
for folder in os.listdir(path):
    for folder2 in os.listdir(path + "/" + folder):
        num_countries += 1

print(num_countries)