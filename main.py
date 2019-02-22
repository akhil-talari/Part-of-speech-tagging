import math
def read_input_data(): #read both train and dev set to develop the model
    fp_train = open('train.conll', 'r', encoding='utf-8')
    fp_dev = open('dev.conll', 'r')
    sentences = []
    input = []
    for line1 in fp_train:
        if(line1 != '\n'):
            sentences.append(line1)
        else:
            input.append(sentences)  # two appends to create a list of list
            sentences = []
    for line2 in fp_dev:
        if (line2 != '\n'):
            sentences.append(line2)
        else:
            input.append(sentences)  # two appends to create a list of list
            sentences = []
    return input

def read_test_data():# Read test.conll and store it in the form of list of lists
    fp_test = open('test.conll', 'r')
    sentences = []
    list_of_ids = []
    sentences_withoutid = []
    test_data = []
    test_data_withoutid =[]
    id_data = []
    for line in fp_test:
        if (line != '\n'):
            line = line.strip('\n')
            content = line.split('\t')
            word = content[0]+content[1]
            id = content[1]
            word_withoutid = content[0]
            sentences.append(word)
            list_of_ids.append(id)
            sentences_withoutid.append(word_withoutid)
        else:
            test_data.append(sentences) #2nd time append to create list of list
            id_data.append(list_of_ids)
            test_data_withoutid.append(sentences_withoutid)
            sentences = []
            list_of_ids = []
            sentences_withoutid = []

    test_data.append(sentences)
    id_data.append(list_of_ids)
    test_data_withoutid.append(sentences_withoutid)
    return test_data, id_data, test_data_withoutid
#----------------------------------------------------------------------------------#
def corpus(doc):#defines word corpus,tag corpus,bigrams_types
    bigrams_types = dict()
    word_corpus = dict()
    tag_corpus = dict()
    for sentence in doc:
            term_count_in_sentence = 0
            for term in sentence:
                term = term.strip('\n')
                content = term.split('\t')
                word = content[0]+content[1]#we concatenate the word with language id
                tag = content[2]

                if tag not in tag_corpus: #tag corpus is a dictionary with tags as keys and their counts as values
                    if(term_count_in_sentence != 0):
                        tag_corpus.update({tag: {'Tot_count':1 }})
                else:
                    tag_corpus[tag]['Tot_count'] += 1
#--------------------------------------------------------------
                if word not in word_corpus: #word corpus has words with their respective tag counts
                    if (term_count_in_sentence != 0):
                        word_corpus.update({word: {'Tags': {tag: {'Count': 1}}, 'Total_count': 1}})
                else:
                    word_corpus[word]['Total_count'] += 1
                    if tag not in word_corpus[word]['Tags']:
                        word_corpus[word]['Tags'].update({tag: {'Count': 1}})
                    else:
                        word_corpus[word]['Tags'][tag]['Count'] += 1
#--------------------------------------------------------------
                if len(sentence) >1 and term_count_in_sentence > 0: #get the bigrams of all the tags
                    first_tag = sentence[term_count_in_sentence - 1].strip('\n').split('\t')[2]
                    bigram = (first_tag, tag)
                    if (bigram not in bigrams_types):
                        bigrams_types.update({bigram: 1})
                    else:
                        bigrams_types[bigram] += 1
                term_count_in_sentence += 1
    return word_corpus, tag_corpus, bigrams_types

def unk_words(word_dict, cutoff):#replace word with 'UNK' for the words with frequency less than or equal to cut-off
    unknown_dict = dict()
    unknown_dict['UNK'] = {'Total_count':0,'Tags':{}}
    for word in word_dict:
        if word_dict[word]['Total_count'] > cutoff:
            unknown_dict[word] = word_dict[word] #unknown_dict same as word_dict
        else:
            total_counts = word_dict[word]['Total_count']
            unknown_dict['UNK']['Total_count'] += total_counts #update count of unknown word
            for tag in word_dict[word]['Tags']:
                if tag not in unknown_dict['UNK']['Tags']:
                    unknown_dict['UNK']['Tags'][tag] = word_dict[word]['Tags'][tag]
                else:
                    unknown_dict['UNK']['Tags'][tag]['Count'] += word_dict[word]['Tags'][tag]['Count']
    return unknown_dict

def Laplace(tag_dict, bigrams_types):#smooth the transition probabilities of the bigram types
    bigram_groups = []
    for tag_1st in tag_dict:
        for tag_2nd in tag_dict:
            bigram_groups.append((tag_1st, tag_2nd))
    for bigram in bigram_groups:
        tag_i = bigram[1]
        tag_i_1 = bigram[0]
        if (bigram in bigrams_types):
            transition_probability = math.log((bigrams_types[bigram]+1) / (tag_dict[tag_i_1]['Tot_count'] + len(tag_dict)))
        else:
            transition_probability = math.log(1 / (tag_dict[tag_i_1]['Tot_count'] + len(tag_dict)))
        if 'P_smoothing' not in tag_dict[tag_i]:
            tag_dict[tag_i].update({'P_smoothing':{tag_i_1:transition_probability}})
        else:
            tag_dict[tag_i]['P_smoothing'].update({tag_i_1:transition_probability})
    return tag_dict

def emission_probabilities(word_dict, tag_dict): #calculate word given tag probability
    for word in word_dict:
        for tag in word_dict[word]['Tags']:
            emission_probability = math.log(word_dict[word]['Tags'][tag]['Count'] / tag_dict[tag]['Tot_count'])
            if 'Prob' not in word_dict[word]['Tags'][tag]:
                word_dict[word]['Tags'][tag].update({'Prob':emission_probability})
            else:
                word_dict[word]['Tags'][tag]['Prob'] = emission_probability
    return word_dict
#-------------------------------------------------------------------------------------------
def viterbi(test_set_sentence, word_dict, tag_dict):#Viterbi algorithm starts here
    sentence =[]
    for word in test_set_sentence:
        if word in word_dict:
            sentence.append(word)
        else:
            sentence.append('UNK')

    for word in word_dict:
        for tag in tag_dict:
            if tag not in word_dict[word]['Tags']:
                word_dict[word]['Tags'].update({tag: {'Count': 0, 'Prob':float('inf')}})

    viterbi_dictionary = dict()
    backpointer = dict()
    for tag in tag_dict:
        viterbi_dictionary.update({tag: {}})
        backpointer.update({tag:{}})
        for i, word in enumerate(sentence):
            viterbi_dictionary[tag].update({(i,word):0.0}) #initialize matrix
            backpointer[tag].update({(i,word):None}) #initialize matrix

    for tag in tag_dict: #start symbol Probability calculation for each sentence
        if len(sentence) != 0:
                if word_dict[sentence[0]]['Tags'][tag]['Prob'] != float('inf'):
                    word_given_tag_probability = word_dict[sentence[0]]['Tags'][tag]['Prob']
                    viterbi_start = 1 +(word_given_tag_probability)
                else:
                    viterbi_start = float('inf')
        if len(sentence) != 0:
            viterbi_dictionary[tag][(0, sentence[0])] = viterbi_start
            backpointer[tag][(0,sentence[0])] = '0'

    for token in range(1,len(sentence)):#Actual probabilities calculation
        for tag in tag_dict:
            column_probability = []
            backpointer_intermediate = []
            if word_dict[sentence[token]]['Tags'][tag]['Prob'] != float('inf'):
                for prev_tag in tag_dict:
                    if viterbi_dictionary[prev_tag][(token-1,sentence[token-1])] != float('inf'):
                        if(tag_dict[tag]['P_smoothing'][prev_tag] != 0.0):
                            column_probability.append(viterbi_dictionary[prev_tag][(token - 1,sentence[token - 1])] +
                                         (tag_dict[tag]['P_smoothing'][prev_tag]) +
                                         (word_dict[sentence[token]]['Tags'][tag]['Prob']))
                            backpointer_intermediate.append((prev_tag,viterbi_dictionary[prev_tag][(token - 1,sentence[token - 1])] + (
                                tag_dict[tag]['P_smoothing'][prev_tag])))
                        else:
                            viterbi_dictionary[tag][(token, sentence[token])] = float('inf')
                            backpointer[tag][(token, sentence[token])] = None

                viterbi_dictionary[tag][(token,sentence[token])] = max(column_probability)
                backpointer[tag][(token,sentence[token])] = max(backpointer_intermediate, key=lambda x: x[1])[0]

            else:
                viterbi_dictionary[tag][(token,sentence[token])] = float('inf')
                backpointer[tag][(token,sentence[token])] = None

    final_pathtrace = []
    for tag in tag_dict:
        if len(sentence) != 0:
            if viterbi_dictionary[tag][(len(sentence)-1,sentence[len(sentence)-1])] != float('inf'):
                final_pathtrace.append((tag,viterbi_dictionary[tag][(len(sentence)-1,sentence[len(sentence)-1])] ))

    final_path = []
    if final_pathtrace:
        end_state = max(final_pathtrace, key=lambda x:x[1])[0]
        path = [end_state]
    for token in range(len(sentence)-1,0,-1):
        tag = backpointer[path[len(path)-1]][(token,sentence[token])]
        path.append(tag)
    for p in range(len(sentence)-1, -1, -1): #loop backwards
         final_path.append(path[p])
    return final_path
#--------------------------------------main---------------------------------------------------
test_data, id_data, test_data_withoutid = read_test_data()
words_corpus, tag_corpus, bigrams_types = corpus(read_input_data())
word_corpus = unk_words(words_corpus, 1)
word_corpus = emission_probabilities(word_corpus, tag_corpus)
#print("word_corpus['soeng']",word_corpus['soeng']) #used for error analysis
tag_corpus = Laplace(tag_corpus, bigrams_types)
fd = open('submission.txt','w') #predicted POS taggers are written here
print("processing...")
for sentence,idd, sentence_withoutid in zip(test_data,id_data,test_data_withoutid):
        part_of_speech_tagging = viterbi(sentence, word_corpus, tag_corpus)
        for index in range(len(sentence)):
            fd.write(sentence_withoutid[index] + '\t' + idd[index] + '\t' + part_of_speech_tagging[index] + '\n')
        fd.write('\n')
print("finished!!!")
fd.close()