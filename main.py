#Setting up the layout body of the main HMM
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from spellchecker import SpellChecker
from viterbi import viterbi, recover_path
from get_probs import counts, probs, emission_probs
from gen_errors import main
from evaluation_measures import accuracy

#train -- list containing the names of the training data files
#test -- list containing the names of the testing data files
#get_corrections returns list of sentences corrected
def get_corrections(train,test):
    #Extract the initial and transition probabilities from training data
    #counts_list is a list of tuples : dictionary probabilities init and transitions
    counts_list = []
    for f in train:
        counts_list.append(counts(f))
    [initial, transitions] = probs(counts_list)

    #Extract emission probabilities from training data by
    #generating list of words and their possible errors
    #err_words is a list of words
    err_words = []
    for w in transitions.keys():
        err_words.append(w)
        #Make sure this is compatible with new function for all errors of a word
        err_words.append(generate_errors(w))
    emissions = emission_probs(err_words, 0.1)

    #Generate errors in the testing data
    #err_data is a list of sentences with errors
    err_data = []
    for f in test:
        err_data.append(main(f))

    #Derive smaller list of hidden states for each sentence to be corrected
    corrected_sentences = []
    spell = SpellChecker()
    for s in err_data:
        #Tokenize the sentence to words
        w = word_tokenize(s)
        #Generate all possible hidden states (corrections) specific to the words in the sentence
        dict = []
        for word in w:
            l = generate_errors(word)
            #Only keep valid english words in the possible hidden states list
            dict = spell.known(l)
        #OR use dict = err_words if we don't consider states for each sentence
        #Run Viterbi on the sentence
        (p,start) = viterbi(transitions,emissions,initial,len(dict),len(w),w,dict)
        #Recover corrected sentence
        corrected_sentences.append(recover_path(len(w), p, dict, start))
    return [corrected_sentences, err_data]

def main():
    #Separating data into testing, development and training data
    raw_data = ["reut2-000.sgm","reut2-001.sgm","reut2-002.sgm","reut2-003.sgm","reut2-004.sgm",
    "reut2-005.sgm","reut2-006.sgm","reut2-007.sgm","reut2-008.sgm","reut2-009.sgm","reut2-010.sgm",
    "reut2-011.sgm","reut2-012.sgm","reut2-013.sgm","reut2-014.sgm","reut2-015.sgm",
    "reut2-016.sgm","reut2-017.sgm","reut2-018.sgm","reut2-019.sgm","reut2-020.sgm","reut2-021.sgm",]
    training_data, test_data = train_test_split(raw_data,train_size=0.75,test_size=0.25,random_state = 0)
    dev_data, testing_data = train_test_split(test_data,train_size=0.5,test_size=0.5,random_state = 0)

    #Getting the corrected data
    [corrected, with_errors] = get_corrections(training_data, dev_data) # -- Training the model
    #corrections = get_corrections(dev_data, test_data) # -- Testing

    #Evaluating results
    acc = accuracy(corrected,with_errors)
