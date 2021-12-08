from tokenize import String
import numpy as np
from nltk import word_tokenize
from numpy.core.fromnumeric import argmax
#Starting the viterbi code from class, 
# then I will incorporate the specific spelling corrections

#transitions : transition probs
#emissions : emission probs
#init : initial probs
#N : Total number of states (number of words in the corpus)
#T : length of the sentence
#s : input sentence
#dict : all words seen in the corpus (i.e all states)
def viterbi(transitions,emissions,init,N,T,s,dict):
    #Tokenize sentence
    print(s)
    w = word_tokenize(s)
    #w = [c for c in s]

    #Create trellis and pointers
    delta = np.zeros((N,T))
    pointers = np.zeros((N,T),dtype=int)

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
            #Store best score and the pointer to previous cell
            delta[j,t] = max(c)
            pointers[j,t] = argmax(c)

    #Return max of last column
    print(delta)
    print(pointers)
    return (pointers, argmax(delta[:,N-1]))

#Recover the best corrected sentence
def recover_path(len,pointers,dict,start):
    path = [dict[start]]
    prev = start
    for i in range(len-1,0,-1):
        index = pointers[prev,i]
        path.append(dict[index])
        prev = index
    path.reverse()
    new_s = ""
    for s in path:
        new_s = new_s + " " + s
    return new_s

#Testing using some small data
pi = {"i" : 0.6, "like" : 0.1, "movies" : 0.3}
a = {
    "i" : {"i" : 0.1, "like" : 0.7, "movies" : 0.2},
    "like" : {"i" : 0.1, "like" : 0.1, "movies" : 0.8},
    "movies" : {"i" : 0.1, "like" : 0.1, "movies" : 0.8},
}
b = {
    "i" : {"i" : 0.7, "liek" : 0.2, "move" : 0.1},
    "like" : {"i" : 0.2, "liek" : 0.6, "move" : 0.2},
    "movies" : {"i" : 0.1, "liek" : 0.1, "move" : 0.8},
}
dict = ["i","like","movies"]
s = "i liek move"
(p,start) = viterbi(a,b,pi,len(dict),3,s,dict)
print(recover_path(3, p, dict, start))