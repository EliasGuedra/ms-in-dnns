import argparse
import sys

choices = ["fibonacci", "square", "triangular", "factorial", "prime"]

def fibonacci(n):

    if n == 0:
        return []
    elif n == 1:
        return[0]

    l = [0, 1]

    for i in range(n-2):
        l.append(l[-1] + l[-2])

    return l

def square(n):
    return [i**2 for i in range(1,n+1)]

def triangular(n):

    l = [0]

    for i in range(1, n+1):
        l.append(l[-1] + i)
    return l [1:]

def factorial(n):

    if n == 0:
        return []

    l = [1]

    for i in range(1, n+1):
        l.append(l[-1]*i)
    return l [1:]

def prime(n):

    if n == 0:
        return []
    
    primes = [2]
    p = 3

    while len(primes) < n:
        d = 2
        while d**2 <= p:
            if p % d == 0:
                break
            d += 1
        else:
            primes.append(p)
        p += 1
    return primes   





def main(args):

    n = args.length
    sequence = args.sequence

    if n <= 0:
        parser.error(
            "Error: --length must be a positive integer"
        )

    if not(sequence in choices):
        parser.error(
            "invalid choice of sequence. The options are fibonacci, square, triangular, factorial, prime."
        )

    if args.sequence == "fibonacci":
        result = fibonacci(n)
    elif args.sequence == "square":
        result = square(n)
    elif args.sequence == "triangular":
        result = triangular(n)
    elif args.sequence == "factorial":
        result = factorial(n)
    elif args.sequence == "prime":
        result = prime(n)
    
    return result




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence",
        help=f"Type of sequence to generate. The options are {','.join(choices)}.",
        default=choices[0]
    )

    parser.add_argument(
        "--length",
        type=int,
        help="Length of the sequence, must be > 0",
        default=10
    )

    args = parser.parse_args()

    print(main(args))