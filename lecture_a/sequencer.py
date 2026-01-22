import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--sequence",
                    choices=["fibonacci", "square", "triangular", "factorial"],
                    required=True)


def fibonacci(n):

    l = [0, 1]

    for i in range(n-1):
        l.append(l[-1] + l[-2])

    return l [1:]


def square(n):
    return [i**2 for i in range(1,n+1)]


def triangular(n):

    l = [0]

    for i in range(1, n+1):
        l.append(l[-1] + i)
    return l [1:]

def factorial(n):
    l = [1]

    for i in range(1, n+1):
        l.append(l[-1]*i)
    return l [1:]

