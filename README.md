# Comp-550
This project contains code to implement a spell checker based on a Hidden Markov Model. The data that this is being trained on are reuters news articles from 1987.
# Main.py
This file contains 2 functions. get_corrections takes 2 arguments, a list of files names of training data, and a list of file names of test data. It calls probs and counts from
get_probs to get the initial and transition probabilites of the words contained in the files with training data. Then it opens a precomputed file with emission probabilities of the 
words, as this is very computationally intensive. After that, it calls generateError from gen_errors to introduce mistakes into the data. It then adds some proper nouns to the dictionary
of correct english words. Then, for all sentences in the test data, viterbi and recover_path are called from viterbi, which runs the viterbi algorithm on the sentence and gives us a corrected 
sentence. Lists of the corrected sentences and sentences with errors are returned. 

The other function, evaluate_results, opens the results from calling get_corrections, does some processing to make sure all the characters are alphanumeric, and then calls evaluate_accuracy
from eval to get the accuracy of the results.

The main method splits the training data into a training and test set, calls get_corrections with these lists as arguments, writes the results of get_corrections, and then calls 
evaluate_results on the data.

# get_probs.py
This file contains 5 functions, emission_probs, get_single_emission_prob, prob_distribution, probs, and counts. 

counts takes one argument, which is the file name of that data that it is going to be run on. It expects the data to be from our data set. The file is opened, then the actual articles
are recovered using beautifulSoup to find the contents of the "body" tags, where the articles are stored. Then each one of these articles are tokenized into sentences, using sent_tokenize 
from nltk. After this, the sentences are tokenized into words, using word_tokenize from nltk. There are 2 dictionaries, one containing the number of times each word came first in a sentence, and another
nested dictionary containing the counts of words that occur after the current word. For example if we had the bigram "he said", d["he"]["said"] would be increased by 1, where d is the nested dictionary.
The words in each sentence are looped through and the counts in these 2 dictionaries are updated at each step. Then a list with these 2 dictionaries as elements, the dictionary of first counts element 0, and
the nested dictionary as element 1, is returned.

probs takes a list of the lists returned from counts as input. It then goes through each element in the list and sums the counts of first and transitions from each element into 2 dictionaries,
1 the sum of first counts and the other the sum of transition counts. Now we have the total number of initial and transition words, so we calculate the initial and transition probabilities. We add
an "UNK" token for when we encounter unknown words in the testing data. We use add one smoothing when calculating the probabilties so that the unknown token does not have 0 probability. Two dictionaries
, one with initial probabilties, and one with transitions are returned.

prob_distribution takes 2 arguments, N which is the number of values we want to generate the distribution for, and error, which is the probabiltiy of there being a typing error. 
A Geometric distribution with 1-error as the first term is generated, and returned in the form of a list, where list[0] is the first term, list[1] the second and so on.

get_single_emission_prob_dist and get_single_emission_prob are very similar. They both take 3 arguments, target, words, and error, where target is the word for which 
we want to calculate the emission probabilties, words is the list of words for which to calculate the emission probabilties for, and error is the probability of making a typing error.
The difference between the 2, is get_single_emission_prob assumes that each word in words is equally likely to be emitted, and get_single_emission_prob_dist calles prob_distribution to
get a distribution and assigns probabilties based on levenshtein distance from target.

emissions takes a list of words and error, and calls get_single_emission_prob on each.

# viterbi.py
This file contains 2 functions, viterbi and recover_path.

viterbi runs the viterbi algorithm given initial, transition, and emission probabilities, as well as a dict of all hidden states, a list of words in the sentence (observed states), 
, the length of the sentence, and the number of hidden states.. After accounting for words not in the dictionaries containing probabilities, the trellis is looped over, and the matrix of pointers
and maximum value of the last column are returned.

recover_path takes 4 arguments, the pointer matrix returned by viterbi, a list of hidden states, the start position in the last column, and the length of the sentence. It then works backwards through
the matrix of pointer to return the reconstructed sentence. It returns the reconstructed sentence as a string.

# gen_errors.py
This file contains 6 functions, generateError, generateAllErrors, typo, characterOrder, missingLetter, and extraLetter.

extraLetter,missingLetter, characterOrder and typo all take a sentence and index as arguments. The index argument is the index of the word for which the error will be generated on.
For all these functions, random indices are chosen to generate the erros, For extraLetter, an extra letter is inserted at the index, for missingLetter a character is removed at the index,
for characterOrder, 2 characters at random indices are swapped, and for typo, a letter is replaced with another random letter at the random index.

generateAllErrors takes a single argument, a word for which all errors will be generated for. Every possible error of each of the types previously mentioned is generated, and this is return as a list of strings. 

generateError takes a sentence, in the form of a string as input. It chooses a random error type and a random word in the sentence. It then calls the correspoinding function to the error type
(extraLetter,missingLetter,characterOrder,typo) on the word and replaces the correct word in the sentence with the incorrect word. It returns the sentence with the error introduced as a string.

# eval.py
This file contains 1 function, evaluate_accuracy. This takes 3 arguments, all lists of strings. These are original, which is the original sentence from the data, w_errors, which is the sentence with
the error added, and corrected, which is our program's attempt at correcting the sentence. All sentences are looped over, and for each word in each sentence, it is checked if the word was correctly modified
by our function (an error was introduced and our model successfully corrected it) , if our function introduced a new error by changing a word that didn't need to be changed, and if our model failed to correct the 
word (if an error was introducted and our model did not fix it to be the same as in the original sentence). The counts of all these metrics are returned by the function.

