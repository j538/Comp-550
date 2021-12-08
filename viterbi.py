import numpy as np
from nltk import word_tokenize
#Starting the viterbi code from class, 
# then I will incorporate the specific spelling corrections

#transitions : transition probs
#emissions : emission probs
#init : initial probs
#N : Total number of states (all words in dict)
#T : length of the sentence
#s : input sentence
#dict : all words seen in the corpus (i.e all states)
def viterbi(transitions,emissions,init,N,T,s,dict):
    #Tokenize sentence
    #w = word_tokenize(s)
    w = [c for c in s]
    print(w)

    #Create trellis and pointers
    delta = np.zeros((N,T))
    pointers = np.zeros((N,T))

    #Initialize first column
    for i in range(N):
        delta[i,0] = init[dict[i]]*emissions[dict[i]][w[0]]

    #Loop over the trellis
    for t in range(1,T):
        for j in range(N):
            c = []
            for i in range(N):
                #Get all the possible scores
                c.append(delta[i,t-1]*transitions[dict[i]][dict[j]]*emissions[dict[j]][w[t]])
            delta[j,t] = max(c)
    #Return max of last column
    print(delta)
    return max(delta[:,N-1])

#Recover the best corrected sentence
def recover_path():
    return

#Testing using some small data
pi = {"X" : 0.2, "Y" : 0.5, "Z" : 0.3}
a = {
    "X" : {"X" : 0.5, "Y" : 0.4, "Z" : 0.1},
    "Y" : {"X" : 0.2, "Y" : 0.3, "Z" : 0.5},
    "Z" : {"X" : 0.1, "Y" : 0.1, "Z" : 0.8},
}
b = {
    "X" : {"!" : 0.1, "@" : 0.9},
    "Y" : {"!" : 0.5, "@" : 0.5},
    "Z" : {"!" : 0.7, "@" : 0.3},
}
dict = ["X","Y","Z"]
s = "!@@!"
print(viterbi(a,b,pi,len(dict),len(s),s,dict))

#How do we access the correct word t. cannot use indices