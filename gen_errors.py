import random
import string
from datetime import datetime

def extraLetter(sentence, index):
    print("Adding a letter in: " + sentence[index])

    word = sentence[index]
    length_word = len(word)

    # Choose random index to add a letter
    random.seed(datetime.now())
    i = random.randint(0, length_word)

    # Generate a random letter
    random_letter = random.choice(string.ascii_letters)

    # Check if i is out of range
    if i == length_word:
        sentence[index] = word + random_letter
    else:
        sentence[index] = word[:i] + random_letter + word[i:]

    return sentence

def missingLetter(sentence, index):
    print("Removing a letter in: " + sentence[index])

    word = sentence[index]
    length_word = len(word) - 1

    # If word only has one letter, choose another random word
    if length_word == 0:
        while length_word == 0:
            random.seed(datetime.now())
            index = random.randint(0, len(sentence) - 1)
            word = sentence[index]
            length_word = len(word) - 1

    # Choose random index to remove a letter
    random.seed(datetime.now())
    i = random.randint(0, length_word)

    # Remove letter
    if i == length_word:
        sentence[index] = word[:i]
    else:
        sentence[index] = word[:i] + word[i + 1:]

    return sentence

def characterOrder(sentence, index):
    print("Modifying the order in: " + sentence[index])

    word = sentence[index]
    length_word = len(word) - 1

    # If word only has one letter, choose another random word
    if length_word == 0:
        while length_word == 0:
            random.seed(datetime.now())
            index = random.randint(0, len(sentence) - 1)
            word = sentence[index]
            length_word = len(word) - 1

    # Choose 2 random indexes to exchange the letters
    random.seed(datetime.now())
    i = random.randint(0, length_word)
    if i == length_word: j = i - 1
    else: j = i + 1

    # Exchange the letters
    if i == length_word:
        sentence[index] = word[:j] + word[i] + word[j]
    else:
        if j == length_word:
            sentence[index] = word[:i] + word[j] + word[i]
        else:
            sentence[index] = word[:i] + word[j] + word[i] + word[j + 1:]

    return sentence

def typo(sentence, index):
    print("Introducing a typo in: " + sentence[index])

    word = sentence[index]
    length_word = len(word) - 1

    # Choose random index to modify a letter
    random.seed(datetime.now())
    i = random.randint(0, length_word)

    # Generate a random letter
    random_letter = random.choice(string.ascii_letters)

    # Introducing the typo
    word = list(word)
    word[i] = random_letter
    sentence[index] = ''.join(word)

    return sentence

"""
    We have 4 possible types of error for a given word that we consider:
        - 0: Extra letter
        - 1: Missing letter
        - 2: Two consecutive characters are reversed
        - 3: Wrong letter (typo)
"""
# Generate 1 or 2 errors per sentence, max of 1 error per word
def generateError(sentence):
    # Last index of the sentence
    sentence = sentence.split()
    length = len(sentence) - 1

    # Pick a random error type
    random.seed(datetime.now())
    error_type = random.randint(0, 3)

    # Pick random word in sentence 
    random.seed(datetime.now())
    index = random.randint(0, length)

    # ERRORS -----------------------------------
    # Extra letter
    if error_type == 0:
        sentence = extraLetter(sentence, index)

    # Missing letter
    elif error_type == 1:
        sentence = missingLetter(sentence, index)

    # Characters order
    elif error_type == 2:
        characterOrder(sentence, index)

    # Typo
    else:
        typo(sentence, index)


    sentence = ' '.join(sentence)
    return sentence


# Read the data from the file and store each sentence in an array cell
def readData(file):
    data = []
    f = open(file, "r")
    for sentence in f:
        data.append(sentence.strip())

    return data


def main():
    data = readData("data.txt")

    data_w_errors = []
    for sentence in data:
        data_w_errors.append(generateError(sentence))

    for s in data_w_errors:
        print(s)


if __name__ == "__main__":
    main()