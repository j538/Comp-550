from bs4 import BeautifulSoup
import collections, nltk, json
from numpy.polynomial import Polynomial
from tqdm import tqdm
#Gets counts of initial words and transitions
#We can call this on all the files we want and then calculate the probabilities
#by looking at all the dictionaries
#Expects file name as input
def counts(data):
    #open file, let beautifulsoup parse it
    f1=open(data,"r")
    data1=f1.read()
    soup1 = BeautifulSoup(data1,'html.parser')
    #get all body tags (this is where articles are)
    words = soup1.findAll('body')
    first = collections.defaultdict(int)
    transitions = collections.defaultdict(lambda: collections.defaultdict(int))
    c = 0
    for w in words:
        #tokenize paragraph into sentences then sentence into words
        sentences = nltk.sent_tokenize(w.text)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            # Check if the word is all alphanumeric and in lowercase
            for word in words:
                word.lower()
                word = ''.join(c for c in word if c.isalnum())
            #add to count of first word
            first[words[0].lower()]+=1
            #go through sentence and add transition occurrences
            for i in range(0,len(words)-1):
                transitions[words[i].lower()][words[i+1].lower()]+=1
            #Adding special end word token in case no words come after
            transitions[words[len(words)-1].lower()]["END_WORD"]+=1
    return [first,transitions]

#Expects list of items returned by counts as input
def probs(counts_list):
    all_first = collections.defaultdict(int)
    all_transitions = collections.defaultdict(lambda: collections.defaultdict(int))
    word_counts_transition=collections.defaultdict(int)
    num_sentences=0
    #compile all first and transition dictionaries into 1
    for pair in counts_list:
        for word in pair[0]:
            all_first[word]+=pair[0][word]
            #record number of sentences to make calculating probability easier
            num_sentences+=pair[0][word]
        for word in pair[1]:
            for word2 in pair[1][word]:
                all_transitions[word][word2]+=pair[1][word][word2]
                #record count of each word to make probabilities easier
                word_counts_transition[word]+=pair[1][word][word2]
    first_probabilities = collections.defaultdict(float)
    transition_probabilities = collections.defaultdict(lambda: collections.defaultdict(float))
    for word in all_first:
        num_initial_words = len(all_first.keys())
        #calculate first probabilities with add one smoothing
        first_probabilities[word] = (all_first[word]+1)/(num_sentences+num_initial_words)
        first_probabilities["UNK"] = 1/(num_sentences+num_initial_words)
    for word in all_transitions:
        num_transition_words = len(all_transitions[word].keys())
        for word2 in all_transitions[word]:
            #calculate transition probabilities with add one smoothing
            transition_probabilities[word][word2] = (all_transitions[word][word2]+1)/(word_counts_transition[word]+num_transition_words)
        transition_probabilities[word]["UNK"] = 1/(word_counts_transition[word]+num_transition_words)
        transition_probabilities["UNK"][word] = 1/(num_transition_words+1)
    transition_probabilities["UNK"]["UNK"] = 1/(num_transition_words+1)
    return [first_probabilities,transition_probabilities]

#generate probability distribution given probability of error and max distance
def prob_distribution(error,N):
    coefficients = []
    #get coefficients for polynomial to solve
    error=1-error
    coefficients.append(error-1)
    for i in range(N):
        coefficients.append(error)
    p = Polynomial(coefficients)
    roots = p.roots()
    #get positive root
    for r in roots:
        if(r.imag==0 and r.real>0):
            coefficient = r.real
    distribution = []
    #calculate distribution for each number
    for i in range(N+1):
        distribution.append(error*pow(coefficient,i))
    return distribution

#returns dictionary of emission probailities for single word (Can use during runtime for unknown)
def get_single_emission_prob(target,words,error):
    emissions=collections.defaultdict(float)
    tmp_distance_counts = collections.defaultdict(list)
    # for each word calculate distance from current word
    max_distance=0
    for word2 in words:
        distance = nltk.edit_distance(target, word2)
        max_distance = max(distance, max_distance)
        tmp_distance_counts[distance].append(word2)
    # get probability distribution
    distribution = prob_distribution(error, max_distance)
    # populate emissions with probabilities
    for dis in tmp_distance_counts:
        for word2 in tmp_distance_counts[dis]:
            emissions[word2] = distribution[dis] / len(tmp_distance_counts[dis])
    return emissions

#calculate emission probability given list of words and a probability of error
def emission_probs(words,error):
    emissions = collections.defaultdict(lambda: collections.defaultdict(float))
    for word in tqdm(words):
        emissions[word]= get_single_emission_prob(word,words,error)
    return emissions

#get_single_emission_prob("test",["testing","toast","testink"],0.1)

""" 
example of getting probabilities and saving to a file
lis = []
lis.append(counts("reut2-000.sgm"))
lis.append(counts("reut2-001.sgm"))

a = probs(lis)
b = emission_probs(a[0].keys(),0.1)
with open("initial_probs_example.json","w") as file:
    file.write(json.dumps(a[0]))
with open("transition_probs_example.json", "w") as file:
    file.write(json.dumps(a[1]))
with open("emission_probs_example.json","w") as file:
    file.write(json.dumps(b))
    """