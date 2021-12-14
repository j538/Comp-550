import numpy as np
from nltk import word_tokenize
from numpy.core.fromnumeric import argmax
from get_probs import emission_probs, get_single_emission_prob

#transitions : dictionary of transition probabilities
#emissions : dictionary of emission probabilities
#init : dictionary of initial probabilities
#N : Total number of hidden states (lenght of dict)
#T : length of the sentence
#w : input words of a sentence
#dict : all hidden states
def viterbi(transitions,emissions,init,N,T,w,dict):
    #Create trellis and pointers
    delta = np.zeros((N,T))
    pointers = np.zeros((N,T),dtype=int)

    #Initialize first column
    for i in range(N):
        #Account for unknown words not stored in the initial probabilities
        initial = init.get(dict[i])
        if initial == None :
            initial = init.get("UNK")

        #Account for unknown words not stored in the emission probabilities
        if emissions.get(dict[i]) == None :
            e_i0 = get_single_emission_prob([dict[i],w[0]], 0.1)
        elif emissions.get(dict[i]).get(w[0]) == None :
            e_i0 = get_single_emission_prob([dict[i],w[0]], 0.1)
        else :
            e_i0 = emissions.get(dict[i]).get(w[0])
        delta[i,0] = initial*e_i0

    #Loop over the trellis
    for t in range(1,T):
        for j in range(N):
            c = []
            #Get all the possible scores
            for i in range(N):
                #Account for unknown words not stored in the transitions
                #Makes sure this structure resprects the UNK token structure
                if transitions.get(dict[i]) == None :
                    t_ij = transitions.get("UNK").get("UNK")
                elif transitions.get(dict[i]).get(dict[j]) == None :
                    t_ij = transitions.get(dict[i]).get("UNK")
                else :
                    t_ij = transitions.get(dict[i]).get(dict[j])

                #Account for unknown words not stored in the emissions
                if emissions.get(dict[j]) == None :
                    e_jt = get_single_emission_prob([dict[j],w[t]], 0.1)
                elif emissions.get(dict[j]).get(w[t]) == None :
                    e_jt = get_single_emission_prob([dict[j],w[t]], 0.1)
                else :
                    e_jt = emissions.get(dict[j]).get(w[t])
                c.append(delta[i,t-1]*t_ij*e_jt)
            #Store best score and the pointer to previous cell
            delta[j,t] = max(c)
            pointers[j,t] = argmax(c)
    #print(delta)
    #Return pointers matrix and max of last column
    return (pointers, argmax(delta[:,N-1]))

#Recover the best corrected sentence
# len : length of the sentence
# pointers : pointer matrix from viterbi
# dict : list of hidden states
# start : start position in the last column
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

"""
Testing using some small data
pi = {"i" : 0.6, "like" : 0.1, "movies" : 0.3}
a = {
    "i" : {"i" : 0.1, "like" : 0.7, "movies" : 0.2},
    "like" : {"i" : 0.1, "like" : 0.1, "movies" : 0.8},
    "movies" : {"i" : 0.1, "like" : 0.1, "movies" : 0.8},
}
b = {
    "i" : {"i" : 0.7, "like" : 0.2, "movy" : 0.1},
    "like" : {"i" : 0.2, "like" : 0.6, "movy" : 0.2},
    "movies" : {"i" : 0.1, "like" : 0.1, "movy" : 0.8},
}
dict = ["i","like","movies"]
s = "i like movy"
(p,start) = viterbi(a,b,pi,len(dict),3,word_tokenize(s),dict)
print(recover_path(3, p, dict, start))
"""