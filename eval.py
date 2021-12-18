from nltk import word_tokenize

def evaluate_accuracy(original, w_errors, corrected):
    correctly_modified = 0
    new_errors = 0 
    failed_corrections = 0
    
    # Tokenize each sentence of each set
    for i in range(len(original)):
        o_words = word_tokenize(original[i])
        e_words = word_tokenize(w_errors[i])
        c_words = word_tokenize(corrected[i])
        
        # Go through each word of sentence i to compare
        for j in range(len(o_words)):
            # correctly_modified: A word was perfectly corrected
            if o_words[j] == c_words[j] and o_words[j] != e_words[j]:
                correctly_modified += 1
            # new_errors: A word was corrected when it didn't need correction
            elif o_words[j] != c_words[j] and o_words[j] == e_words[j]:
                new_errors += 1
            # failed_corrections: A word with errors was left uncorrected or was wrongly corrected
            elif o_words[j] != c_words[j] and o_words[j] != e_words[j]:
                failed_corrections += 1

    return correctly_modified, new_errors, failed_corrections
