#Setting up the layout body of the main HMM
from nltk import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from spellchecker import SpellChecker
from bs4 import BeautifulSoup
import json, collections, tqdm
from eval import evaluate_accuracy
from viterbi import viterbi, recover_path
from get_probs import counts, probs
from gen_errors import generateAllErrors, generateError

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

    #Extract emission probabilities from training data
    print("Extracting emission probabilities.")
    #err_words = []
    #all_words = [t for t in transitions.keys()]
    #emissions = collections.defaultdict(lambda: collections.defaultdict(float))
    #for w in all_words:
    #    tmp_list = generateAllErrors(w)
        #rint = random.randint(0, len(tmp_list)-1)
    #    emissions[w] = get_single_emission_prob(w,all_words+tmp_list,0.1)

    #Using precomputed emissions since too long to compute them every time
    emissions = collections.defaultdict(lambda: collections.defaultdict(float))
    with open("emission_probs_v2.json","r") as file:
       emissions = json.load(file)

    #This was used to unzip the emissions
    #from zipfile import ZipFile
    #with ZipFile("full_emissions.zip","r") as zip_file:
        #tmp = zip_file.read("full_emissions.json")
    #    with zip_file.open("full_emissions.json") as file:
    #        f = file.read()
    #        emissions = json.loads(f.decode("utf-8"))

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
                if len(word_tokenize(sentence)) >= 5:
                    sentence = ' '.join(c.lower() for c in word_tokenize(sentence) if c.isalnum())
                    err_data.append(generateError(sentence))

    #Derive smaller list of hidden states for each sentence to be corrected
    print("Selecting all hidden states.")
    corrected_sentences = []
    spell = SpellChecker()
    spell.word_frequency.load_words("reuter")
    tmp=[]
    #Add proper nouns in data to spellchecker
    with open("all-places-strings.lc.txt") as file:
        tmp = file.readlines()
    for line in tmp:
        spell.word_frequency.load_words(line.split())
    with open("all-exchanges-strings.lc.txt") as file:
        tmp = file.readlines()
    for line in tmp:
        spell.word_frequency.load_words(line.split())
    with open("all-orgs-strings.lc.txt") as file:
        tmp = file.readlines()
    for line in tmp:
        spell.word_frequency.load_words(line.split())
    with open("all-people-strings.lc.txt") as file:
        tmp = file.readlines()
    for line in tmp:
        spell.word_frequency.load_words(line.split())
    with open("all-topics-strings.lc.txt") as file:
        tmp = file.readlines()
    for line in tmp:
        spell.word_frequency.load_words(line.split())
    spell.word_frequency.load_words(tmp)
    
    for s in tqdm.tqdm(err_data):
        #Tokenize the sentence to words and remove non letter characters
        w = [word.lower() for word in word_tokenize(s)]
        for word in w :
            if not word.isalnum() :
                w.remove(word)

        #Generate all possible hidden states (corrections) specific to the words in the sentence
        all_states = []
        for word in w:
            if len(spell.unknown([word])) == 1:
                l = generateAllErrors(word)
                wrong_words = spell.unknown(l)
                for n in l:
                    if not n in wrong_words:
                        all_states.append(n)
            else :
                all_states.append(word)

        #Run Viterbi on the sentence
        #print("-------------------------------------------------------------------------------------------")
        #print(f"Running Viterbi on :  {w}")
        length_sentence = len(w)
        (p,start) = viterbi(transitions,emissions,initial,len(all_states),length_sentence,w,all_states)
        
        #Recover corrected sentence
        new_sentence = recover_path(length_sentence, p, all_states, start)
        corrected_sentences.append(new_sentence)
        #print("Got sentence : " + new_sentence)
    return [corrected_sentences, err_data]

def main():
    #Separating data into testing and training data
    raw_data = ["reut2-000.sgm","reut2-001.sgm","reut2-002.sgm","reut2-003.sgm","reut2-004.sgm",
    "reut2-005.sgm","reut2-006.sgm","reut2-007.sgm","reut2-008.sgm","reut2-009.sgm","reut2-010.sgm",
    "reut2-011.sgm","reut2-012.sgm","reut2-013.sgm","reut2-014.sgm","reut2-015.sgm",
    "reut2-016.sgm","reut2-017.sgm","reut2-018.sgm","reut2-019.sgm","reut2-020.sgm","reut2-021.sgm",]
    training_data, test_data = train_test_split(raw_data,train_size=0.75,test_size=0.25,random_state = 0)

    #Getting the corrected data
    #[corrected, with_errors] = get_corrections(training_data,["reut2-021.sgm"]) # -- Training the model
    #[corrected, with_errors] = get_corrections(["reut2-021.sgm"],["reut2-021.sgm"]) #--Training on small dataset
    #[corrected, with_errors] = get_corrections(training_data, test_data) # -- Final testing

    #Write the obtained data to a file so that we only have to run it once
    #with open("corrected_data_error_type0_v2.json","w") as file:
    #   file.write(json.dumps(corrected))
    #with open("with_errors_type0_v2.json","w") as file:
    #    file.write(json.dumps(with_errors))
    #Evaluating results
    evaluate_results(test_data)

def evaluate_results(test):
    test_data_alphanumeric = []
    #Chg test_data to alphanumeric and we remove sentences < 5 
    for t in test:
        f = open(t,"r")
        data = f.read()
        soup = BeautifulSoup(data,'html.parser')
        body = soup.findAll('body')
        for b in body:
            sentences = sent_tokenize(b.text)
            for sentence in sentences:
                sentence = word_tokenize(sentence)
                if len(sentence) >= 5:
                    for i in range(len(sentence)):
                        sentence[i] = sentence[i].lower()
                        sentence[i] = ''.join(c for c in sentence[i] if c.isalnum())
                    sentence = ' '.join(sentence)
                    test_data_alphanumeric.append(sentence)

    #Access the error data for all types of errors
    with open("with_errors.txt","r") as file:
        with_errors = json.load(file)
    with open("with_errors_type0_v2.json","r") as file:
        with_errors0 = json.load(file)
    with open("with_errors_type1_v2.json","r") as file:
        with_errors1 = json.load(file)
    with open("with_errors_type2_v2.json","r") as file:
        with_errors2 = json.load(file)
    with open("with_errors_type3_v2.json","r") as file:
        with_errors3 = json.load(file)

    #Change the error data to alphanumeric
    err_data = [] 
    for s in with_errors:
        words = s.split()
        for i in range(len(words)):
            words[i] = words[i].lower()
            words[i] = ''.join(c for c in words[i] if c.isalnum())
        words = ' '.join(words)
        err_data.append(words)
    err_data0 = []
    for s in with_errors0:
        words = s.split()
        for i in range(len(words)):
            words[i].lower()
            words[i] = ''.join(c for c in words[i] if c.isalnum())
        words = ' '.join(words)
        err_data0.append(words)
    err_data1 = []
    for s in with_errors1:
        words = s.split()
        for i in range(len(words)):
            words[i].lower()
            words[i] = ''.join(c for c in words[i] if c.isalnum())
        words = ' '.join(words)
        err_data1.append(words)
    err_data2 = []
    for s in with_errors2:
        words = s.split()
        for i in range(len(words)):
            words[i].lower()
            words[i] = ''.join(c for c in words[i] if c.isalnum())
        words = ' '.join(words)
        err_data2.append(words)
    err_data3 = []
    for s in with_errors3:
        words = s.split()
        for i in range(len(words)):
            words[i].lower()
            words[i] = ''.join(c for c in words[i] if c.isalnum())
        words = ' '.join(words)
        err_data3.append(words)

    #Access the corrected data
    with open("corrected_data.txt","r") as f:
        corrected = json.load(f)
    with open("corrected_data_error_type0_v2.json","r") as f:
        corrected_0 = json.load(f)
    with open("corrected_data_error_type1_v2.json","r") as f:
        corrected_1 = json.load(f)
    with open("corrected_data_error_type2_v2.json","r") as f:
        corrected_2 = json.load(f)
    with open("corrected_data_error_type3_v2.json","r") as f:
        corrected_3 = json.load(f)

    print("Getting all eval stats")
    correct, new_errors, failed = evaluate_accuracy(test_data_alphanumeric,err_data,corrected)
    correct0, new_errors0, failed0 = evaluate_accuracy(test_data_alphanumeric,err_data0,corrected_0)
    correct1, new_errors1, failed1 = evaluate_accuracy(test_data_alphanumeric,err_data1,corrected_1)
    correct2, new_errors2, failed2 = evaluate_accuracy(test_data_alphanumeric,err_data2,corrected_2)
    correct3, new_errors3, failed3 = evaluate_accuracy(test_data_alphanumeric,err_data3,corrected_3)

    # Final results 
    print("Overall performance all mistakes : ")
    print(f"correctly modified : {correct}, new errors introduced : {new_errors}, failed correcting : {failed}")
    print("Overall performance extra letter : ")
    print(f"correctly modified : {correct0}, new errors introduced : {new_errors0}, failed correcting : {failed0}")
    print("Overall performance missing letter : ")
    print(f"correctly modified : {correct1}, new errors introduced : {new_errors1}, failed correcting : {failed1}")
    print("Overall performance consecutive characters reversed: ")
    print(f"correctly modified : {correct2}, new errors introduced : {new_errors2}, failed correcting : {failed2}")
    print("Overall performance typo : ")
    print(f"correctly modified : {correct3}, new errors introduced : {new_errors3}, failed correcting : {failed3}")


if __name__ == "__main__":
    main()