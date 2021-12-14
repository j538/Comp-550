#Setting up the layout body of the main HMM
from nltk import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from spellchecker import SpellChecker
from bs4 import BeautifulSoup
import json, random, collections
from viterbi import viterbi, recover_path
from get_probs import counts, probs, emission_probs
from gen_errors import generateAllErrors, generateError
from evaluation_measures import accuracy

#train -- list containing the names of the training data files
#test -- list containing the names of the testing data files
#get_corrections returns list of sentences corrected
def get_corrections(train,test):
    #Extract the initial and transition probabilities from training data
    #counts_list is a list of tuples : dictionary probabilities init and transitions
    print("Extracting initial and tranisition probabilities.")
    counts_list = []
    for f in train:
        counts_list.append(counts(f))
    [initial, transitions] = probs(counts_list)

    #Extract emission probabilities from training data by
    #generating list of words and their possible errors
    #err_words is a list of words
    print("Extracting emission probabilities.")
    err_words = []
    #for w in transitions.keys():
    #    err_words.append(w)
    #    tmp_list = generateAllErrors(w)
        #rint = random.randint(0, len(tmp_list)-1)
    #    for t in tmp_list:
    #        err_words.append(t)
    #emissions = emission_probs(err_words, 0.1)

    #Using precomputed emissions since too long to compute them
    emissions = collections.defaultdict(lambda: collections.defaultdict(float))
    with open("emission_probs_example_21.json","r") as file:
       emissions = json.load(file)

    #Generate errors in the testing data
    #err_data is a list of sentences with errors
    print("Introducing mistakes in the data.")
    err_data = []
    for f in test:
        f1=open(f,"r")
        data1=f1.read()
        soup1 = BeautifulSoup(data1,'html.parser')
        #get all body tags (this is where articles are)
        words = soup1.findAll('body')
        for w in words:
            #tokenize paragraph into sentences
            sentences = sent_tokenize(w.text)
            for sentence in sentences:
                err_data.append(generateError(sentence))

    #Derive smaller list of hidden states for each sentence to be corrected
    print("Selecting all hidden states.")
    corrected_sentences = []
    spell = SpellChecker()
    for s in err_data:
        #Tokenize the sentence to words
        w = word_tokenize(s)
        #Generate all possible hidden states (corrections) specific to the words in the sentence
        all_states = []
        for word in w:
            l = generateAllErrors(word)
            #Only keep valid english words in the possible hidden states list
            #Make sure this chooses the correct words only
            known = spell.known(l)
            #print(known)
            for n in known:
                all_states.append(n)
        #print(all_states)
        #OR use dict = err_words if we don't consider states for each sentence
        #Run Viterbi on the sentence
        print("Running Viterbi.")
        (p,start) = viterbi(transitions,emissions,initial,len(all_states),len(w),w,all_states)
        #Recover corrected sentence
        print("Recovering final sentences.")
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
    #[corrected, with_errors] = get_corrections(training_data, dev_data) # -- Training the model
    #corrections = get_corrections(dev_data, test_data) # -- Testing

    #Testing on some small data set
    [corrected, with_errors] = get_corrections(["reut2-021.sgm"], ["reut2-021.sgm"])
    #with open("corrected_data.json","w") as file:
    #   file.write(json.dumps(corrected))

    #Evaluating results
    acc = accuracy(corrected,with_errors)
    print(acc)

if __name__ == "__main__":
    main()