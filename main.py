#Setting up the layout body of the main HMM
from nltk import word_tokenize, sent_tokenize
import enchant #Needs to be downlaoded I think
from viterbi import viterbi, recover_path
from get_probs import counts, probs

def main(dataset):
    #Preprocessing data -- train, test split, to lower case, remove weird characters

    #Extract the initial and transition probabilities from data with no errors
    counts_list = counts(dataset)
    [initial, transitions] = probs(counts_list)

    #Generate errors in the data
    err_data = Error_fct(dataset)

    #Extract emission probabilities from data with errors
    emissions = Emission_fct(err_data)

    #Tokenize the data into sentences
    sentences = sent_tokenize(err_data)

    #Initialize empty output data
    corrected_sentences = []

    #Run viterbi on sentences that have misspells
    for s in sentences:
        #Check if the sentence has some misspelled word?
        #--

        #Tokenize the sentence to words
        w = word_tokenize(s)

        #call the Error function on each word in w to generate all possible hidden states (corrections)
        #OR derive a list of all different words in the data = dict
        dict = []
        for word in w:
            l = Error_fct(word)
            for word_corrected in l:
                dict.append(word_corrected)
        #Only keep valid english words in the possible hidden states list
        for word in dict:
            d = enchant.Dict("en_US")
            if not d.check(word) :
                dict.remove(word)

        #Run viterbi
        (p,start) = viterbi(transitions,emissions,initial,len(dict),len(w),w,dict)
        corrected_sentences.append(recover_path(len(w), p, dict, start))

    #Using corrected_sentences, perfomr evlaution measures here