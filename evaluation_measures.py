#Evaluation measures
from nltk import word_tokenize
from spellchecker import SpellChecker

#number of corrected mistakes / total number of mistakes
#This only checks spelling and does not account for the sense of the corrected words
def accuracy(corrected, with_errors):
    num = 0
    denom = 0
    spell = SpellChecker()
    for i in range(len(corrected)):
        #Tokenize sentence into list of words
        s_w = word_tokenize(corrected[i])
        t_w = word_tokenize(with_errors[i])
        #print(s_w,t_w)
        #Compute number of mistakes in sentence before and after correction
        num_mistakes = len(spell.unknown(t_w))
        num_mistakes_after = len(spell.unknown(s_w))
        #Update numerator and denominator
        num += num_mistakes - num_mistakes_after #if this is negative : introduced new errors
        #print(num_mistakes)
        denom += num_mistakes
    return num/denom

