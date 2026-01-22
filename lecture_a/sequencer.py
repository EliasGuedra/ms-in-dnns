import argparse

parser = argparse.ArgumentParser()
parser.parse_args()



def fibinacci(n):

    l = [0, 1]

    for i in range(n-1):
        l.append(l[-1] + l[-2])

    return l [1:]


def prime_numbers(n):

    l = []


