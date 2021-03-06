import numpy as np
from numpy.core.fromnumeric import argmax
from gen_errors import generateAllErrors
from get_probs import get_single_emission_prob

#Running the Viterbi algorithm
#transitions : dictionary of transition probabilities
#emissions : dictionary of emission probabilities
#init : dictionary of initial probabilities
#N : Total number of hidden states (lenght of dict)
#T : length of the sentence
#w : array of words of a sentence
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
            tmp_w = generateAllErrors(dict[i])
            tmp_w.append(dict[i])
            new_emissions = get_single_emission_prob(dict[i], tmp_w, 0.1)
            emissions[dict[i]] = new_emissions
            
        if emissions.get(dict[i]).get(w[0]) == None :
            words_in_emission = emissions.get(dict[i]).keys()
            new_words =[]
            for word in w :
                if not word in words_in_emission:
                    new_words.append(word)
            new_emissions = get_single_emission_prob(dict[i], new_words, 0.1)
            for key in new_emissions:
                emissions[dict[i]][key] = new_emissions[key]
        #Compute trellis value for first column
        e_i0 = emissions.get(dict[i]).get(w[0])
        delta[i,0] = initial*e_i0

    #Loop over the trellis
    for t in range(1,T):
        for j in range(N):
            c = []
            #Get all the possible scores
            for i in range(N):
                #Account for unknown words not stored in the transitions
                if transitions.get(dict[i]) == None :
                    if transitions.get("UNK").get(dict[j]) == None:
                        t_ij = transitions.get("UNK").get("UNK")
                    else:
                        t_ij = transitions.get("UNK").get(dict[j])
                elif transitions.get(dict[i]).get(dict[j]) == None :
                    t_ij = transitions.get(dict[i]).get("UNK")
                else :
                    t_ij = transitions.get(dict[i]).get(dict[j])

                #Account for unknown words not stored in the emissions
                if emissions.get(dict[j]) == None :
                    tmp_w = generateAllErrors(dict[i])
                    tmp_w.append(dict[i])
                    new_emissions = get_single_emission_prob(dict[j], tmp_w, 0.1)
                    emissions[dict[j]] = new_emissions

                if emissions.get(dict[j]).get(w[t]) == None :
                    words_in_emission = emissions.get(dict[j]).keys()
                    new_words =[]
                    for word in w :
                        if not word in words_in_emission:
                            new_words.append(word)
                    new_emissions = get_single_emission_prob(dict[j], new_words, 0.1)
                    for key in new_emissions:
                        emissions[dict[j]][key] = new_emissions[key]
                #Store all values for one cell
                e_jt = emissions.get(dict[j]).get(w[t])
                c.append(delta[i,t-1]*t_ij*e_jt)
            #Store best score of this cell and the pointer to previous cell
            delta[j,t] = max(c)
            pointers[j,t] = argmax(c)
    #Return pointers matrix and max of last column
    return (pointers, argmax(delta[:,T-1]))

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
