"""
Python Basics
"""
from numpy.random import normal


def basics():
    """
    basic function
    """
    list_of_numbers = [1, 2, 3, 4, 5, 6]
    for number in list_of_numbers:
        print(number)
        if number % 2 == 0:
            print("is even")
        else:
            print("is odd")
    print("All done.")
    anormal = normal(2.5, 5.0, 10)
    print(anormal)
    xlist = [1, 2, 3, 4, 5, 6]
    print(len(xlist))
    print(xlist[:3])
    print(xlist[3:])
    print(xlist[-2:])
    xlist.extend([7, 8])
    print(xlist)
    xlist.append(9)
    print(xlist)
    ylist = [10, 11, 12]
    list_of_lists = [xlist, ylist]
    print(list_of_lists)
    print(ylist[1])
    zlist = [3, 2, 1]
    zlist.sort()
    print(zlist)
    zlist.sort(reverse=True)
    print(zlist)
    xtuple = (1, 2, 3)
    print(len(xtuple))
    ytuple = (4, 5, 6)
    print(ytuple[2])
    list_of_tuples = [xtuple, ytuple]
    print(list_of_tuples)
    (age, income) = "32,120000".split(",")
    print(age)
    print(income)


def dictionary():
    """
    dictionary function
    """
    captains = {}
    captains["Enterprise"] = "Kirk"
    captains["Enterprise D"] = "Picard"
    captains["Deep Space Nine"] = "Sisko"
    captains["Voyager"] = "Janeway"
    print(captains["Voyager"])
    print(captains.get("Enterprise"))
    print(captains.get("NX-01"))
    for key, value in captains.items():
        print(key + ": " + value)
    square_it = lambda x: x * x
    print(square_it(2))
    do_something = lambda f, x: f(x)
    print(do_something(square_it, 3))
    print(do_something(lambda x: x * x * x, 3))
    print(1 == 3)
    print(True or False)
    if 1 == 3:
        print("How did that happen?")
    elif 1 > 3:
        print("Yikes")
    else:
        print("All is well with the world")
    for xnum in range(10):
        print(xnum)
    for xnum in range(10):
        if xnum == 1:
            continue
        if xnum > 5:
            break
        print(xnum)
    xnum = 0
    while xnum < 10:
        print(xnum)
        xnum += 1


if __name__ == "__main__":
    basics()
    dictionary()
