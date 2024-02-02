i = 0
with open('../data/ch4/type1.ch4.1.gro') as file:
    for line in file:
        if i > 10:
            break
        print(line)
        i += 1

    # headers = [line.rstrip() for line in file if 'step=' in line]
    # steps = [int(h.split('=')[-1]) for h in headers]


# indexes = indexes = [25000 + 250000 * i * 65 for i in range(100)] + [1640025000]
# for i in indexes:
#     if i not in steps:
#         print(i)
