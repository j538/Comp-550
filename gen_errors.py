import random
import string
from datetime import datetime

def extraLetter(sentence, index):
    word = sentence[index]
    length_word = len(word)

    # Choose random index to add a letter
    random.seed(datetime.now())
    i = random.randint(0, length_word)

    # Generate a random letter
    random_letter = random.choice(string.ascii_lowercase)

    # Check if i is out of range
    if i == length_word:
        sentence[index] = word + random_letter
    else:
        sentence[index] = word[:i] + random_letter + word[i:]

    return sentence

def missingLetter(sentence, index):
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
    word = sentence[index]
    length_word = len(word) - 1

    # Choose random index to modify a letter
    random.seed(datetime.now())
    i = random.randint(0, length_word)

    # Generate a random letter
    random_letter = random.choice(string.ascii_lowercase)

    # Introducing the typo
    word = list(word)
    word[i] = random_letter
    sentence[index] = ''.join(word)

    return sentence

# Generate all the possible errors of a given word
def generateAllErrors(word):
    word_variations = []
    
    # Go through every possible index of the word
    last_index = len(word) - 1
    letters = list(string.ascii_lowercase)
    
    # Extra letter
    for i in range(last_index + 2):
        if i < last_index + 1:
            for j in range(26):
                word_variations.append(word[:i] + letters[j] + word[i:])
        else:
            for j in range(26):
                word_variations.append(word + letters[j])
    
    # Missing letter
    for i in range(last_index + 1):
        if i == last_index:
            word_variations.append(word[:i])
        else:
            word_variations.append(word[:i] + word[i + 1:])
    

    # Character order
    for i in range(last_index):
        j = i + 1
        if j == last_index:
            word_variations.append(word[:i] + word[j] + word[i])
        else:
            word_variations.append(word[:i] + word[j] + word[i] + word[j + 1:])

    # Typo
    for i in range(last_index + 1):
        for j in range(26):
            letter = letters[j]
            if letter != word[i]: 
                word_variations.append(word[:i] + letter + word[i + 1:])

    return word_variations

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
