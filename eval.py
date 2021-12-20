from nltk import word_tokenize
from spellchecker import SpellChecker
from collections import Counter

def findMissingWords(o_words, c_words, smallest_length):
    for i in range(smallest_length):
        if o_words[i] != c_words[i]:
            word1 = set(o_words[i])
            word2 = set(c_words[i])
            diff = word1.symmetric_difference(word2)
            if len(diff) > 1:
                return o_words[i]

    return o_words[len(o_words) - 1]

def evaluate_accuracy(original, w_errors, corrected):
    correctly_modified = 0
    new_errors = 0 
    failed_corrections = 0
    total_number_words = 0

    # Tokenize each sentence of each set
    for i in range(len(original)):
        o_words = original[i].split()
        e_words = w_errors[i].split()
        c_words = corrected[i].split()
        
        length_o = len(o_words)
        length_e = len(e_words)
        length_c = len(c_words)

        if length_o == length_c and length_o == length_e:
            for j in range(length_o):
                total_number_words += 1
                # correctly_modified: A word was perfectly corrected
                if o_words[j] == c_words[j] and o_words[j] != e_words[j]:
                    correctly_modified += 1
                # new_errors: A word was corrected when it didn't need correction
                elif o_words[j] != c_words[j] and o_words[j] == e_words[j]:
                    new_errors += 1
                # failed_corrections: A word with errors was left uncorrected or was wrongly corrected
                elif o_words[j] != c_words[j] and o_words[j] != e_words[j]:
                    #print("original: " + str(o_words[j]))
                    #print("errors: " + str(e_words[j]))
                    #print("corrected: " + str(c_words[j]))
                    failed_corrections += 1
        
    return correctly_modified, new_errors, failed_corrections, total_number_words
