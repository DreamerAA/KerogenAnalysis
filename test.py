# name = '../data/ch4/type1.ch4.1.gro'
name = '../data/h2/type1.h2.1.gro'

# i = 0
with open(name) as file:
    #     for line in file:
    #         if i > 10:
    #             break
    #         print(line)
    #         i += 1

    headers = [line.rstrip() for line in file if 'step=' in line]
    steps = [int(h.split('=')[-1]) for h in headers]
    print(len(steps), steps[0], steps[-1])

# indexes = [25000 + 250000 * i * 66 for i in range(100)] + [1653025000]
# for i in indexes:
#     if i not in steps:
#         print(i)
