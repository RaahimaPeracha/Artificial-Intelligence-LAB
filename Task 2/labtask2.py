import random
print("Welcome to FizzBuzz!!!")

while True:
    num = random.randint (1,100)
    print("The Number is:", num)
    if num % 5 == 0 and num % 3 == 0:
        print("FizzBuzz")
    elif num % 5 == 0:
        print("Buzz")
    elif  num % 3 == 0:
        print("Fizz")
    else:
        print("You are eliminated!!!")
        break