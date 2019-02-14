import numpy as np
import math
import time

K_FACTOR = 0.1
GOOD_TURING_FACTOR = 20
UNKNOWN_RATE = 0.6

'''
The method generates an unsmoothed unigram model, given a file name
@param filename: the training corpus's file name
@param log_space_flag: shows whether or not the log space is used
return the generated unigram model
'''
def generate_uni_model_unsmoothed(filename, log_space_flag = 0):
	corpus = open(filename,'r')
	corpus_list = corpus.read().replace("\n", " ").split(" ")
	corpus_dict = {}
	for x in corpus_list:
	    if x in corpus_dict:
	        corpus_dict[x] = corpus_dict.get(x) + 1
	    else:
	        corpus_dict[x] = 1
	corpus_unigram = {}
	sum_of_list = sum(corpus_dict.values())

	for i in corpus_dict.keys():
		if log_space_flag:
			corpus_unigram[i] = math.log(corpus_dict[i]) - math.log(sum_of_list)
		else:
			corpus_unigram[i] = corpus_dict[i]/float(sum_of_list)
	#print(sum(corpus_unigram.values()))
	return corpus_unigram

'''
The method generates a unigram model, given a file name
@param filename: the training corpus's file name
@param smoothing: smoothing method, 1:add-k, 2:good-turing, 3:k/n
return the generated unigram model
'''
def generate_uni_model(filename, smoothing = 1):
	corpus = open(filename,'r')
	corpus_list = corpus.read().replace("\n", " ").split(" ")
	corpus_dict = {}
	for x in corpus_list:
		if x in corpus_dict:
			corpus_dict[x] = corpus_dict.get(x) + 1
		else:
			corpus_dict[x] = 1

	for i in range(len(corpus_list)):
		if corpus_dict[corpus_list[i]] == 1:
			corpus_list[i] = '<UNK>'

	corpus_dict = {}
	for x in corpus_list:
		if x in corpus_dict:
			corpus_dict[x] = corpus_dict.get(x) + 1
		else:
			corpus_dict[x] = 1

	corpus_unigram = {}
	sum_of_list = sum(corpus_dict.values())

	freq_counter = [0] * (GOOD_TURING_FACTOR+1)
	for k in corpus_dict.keys():
		if corpus_dict[k] < (GOOD_TURING_FACTOR+1):
			freq_counter[corpus_dict[k]] += 1

	for i in corpus_dict.keys():
		if smoothing == 0: # DO NOT USE THIS. FOR UNSMOOTHED MODEL, USE THE ABOVE METHOD
			corpus_unigram[i] = math.log(corpus_dict[i]) - math.log(sum_of_list)
		elif smoothing == 1:
			corpus_unigram[i] = math.log(corpus_dict[i]+K_FACTOR) - math.log(sum_of_list + K_FACTOR*len(corpus_dict))
		elif smoothing == 2:
			if corpus_dict[i] < GOOD_TURING_FACTOR:
				corpus_unigram[i] = math.log((corpus_dict[i]+1)*freq_counter[corpus_dict[i]+1]) - math.log(freq_counter[corpus_dict[i]])
			else: 
				corpus_unigram[i] = math.log(corpus_dict[i]) - math.log(sum_of_list)
		elif smoothing == 3:
			if corpus_dict[i] > 1: 
				corpus_unigram[i] = math.log(corpus_dict[i] - 0.75) - math.log(sum_of_list)
			elif corpus_bigram[k_1][k_2] == 1: 
				corpus_unigram[i] = math.log(corpus_dict[i] - 0.5) - math.log(sum_of_list)
			else:
				corpus_unigram[i] = 0.000027

	return corpus_unigram

'''
The method generates an unsmoothed bigram model, given a file name
@param filename: the training corpus's file name
return the generated bigram model
'''
def generate_bi_model_unsmoothed(filename, log_space_flag = 0):
 	global K_FACTOR
 	corpus = open(filename,'r')
 	corpus_list = corpus.read().replace("\n", " ").replace(".", ". </s> <s>").replace("!", "! </s> <s>").replace("?", "? </s> <s>").split(" ")
 	corpus_dict = {}
	for x in corpus_list:
		if x in corpus_dict:
			corpus_dict[x] = corpus_dict.get(x) + 1
		else:
			corpus_dict[x] = 1

	for i in range(len(corpus_list)):
		if corpus_dict[corpus_list[i]] == 1:
			prob = np.random.uniform(0,1)
			if prob < UNKNOWN_RATE:
				corpus_list[i] = '<UNK>'

 	corpus_dict = {}
 	for x in corpus_list:
 	    if x in corpus_dict:
 	        corpus_dict[x] = corpus_dict.get(x) + 1
 	    else:
 	        corpus_dict[x] = 1
 	corpus_bigram = {}
 	for i in corpus_list:
 		if i not in corpus_bigram:
 			corpus_bigram[i] = {}

 	for i in range(len(corpus_list)-1):
 		curr_token = corpus_list[i]
 		next_token = corpus_list[i+1]
 		if next_token in corpus_bigram[curr_token]:
 			corpus_bigram[curr_token][next_token] += 1
 		else:
 			corpus_bigram[curr_token][next_token] = 1

 	for k_1 in corpus_bigram.keys():
 		for k_2 in corpus_bigram[k_1].keys():
 			if (log_space_flag) :
				corpus_bigram[k_1][k_2] = math.log(corpus_bigram[k_1][k_2]) - math.log(corpus_dict[k_1])
 			else:
 				corpus_bigram[k_1][k_2] = corpus_bigram[k_1][k_2]/float(corpus_dict[k_1])
 	return corpus_bigram
'''
The method generates a bigram model, given a file name
@param filename: the training corpus's file name
@param smoothing: 0:unsmoothed, 1:add-k smooth, 2:good-turing, 3: k/n
return the generated bigram model
'''
def generate_bi_model(filename, smoothing=1):
	corpus = open(filename,'r')
	corpus_list = corpus.read().replace("\n", " ").split(" ")
	corpus_dict = {}
	for x in corpus_list:
		if x in corpus_dict:
			corpus_dict[x] = corpus_dict.get(x) + 1
		else:
			corpus_dict[x] = 1

	for i in range(len(corpus_list)):
		if corpus_dict[corpus_list[i]] == 1:
			prob = np.random.uniform(0,1)
			if prob < UNKNOWN_RATE:
				corpus_list[i] = '<UNK>'

	corpus_dict = {}
	for x in corpus_list:
		if x in corpus_dict:
			corpus_dict[x] = corpus_dict.get(x) + 1
		else:
			corpus_dict[x] = 1
			
	corpus_bigram = {}
	for key_1 in corpus_dict.keys():
		corpus_bigram[key_1] = {}
		for key_2 in corpus_dict.keys():
			corpus_bigram[key_1][key_2] = 0

	for i in range(len(corpus_list)-1):
		curr_token = corpus_list[i]
		next_token = corpus_list[i+1]
		corpus_bigram[curr_token][next_token] += 1

	freq_counter = [0] * (GOOD_TURING_FACTOR+1)
	for k_1 in corpus_bigram.keys():
		for k_2 in corpus_bigram[k_1].keys():
			if corpus_bigram[k_1][k_2] < (GOOD_TURING_FACTOR+1):
				freq_counter[corpus_bigram[k_1][k_2]] += 1

	for k_1 in corpus_bigram.keys():
		for k_2 in corpus_bigram[k_1].keys():
			if smoothing == 0: # DO NOT USE THIS. FOR UNSMOOTHED MODEL, USE THE ABOVE METHOD
				if corpus_bigram[k_1][k_2] is not 0:
					corpus_bigram[k_1][k_2] = math.log(corpus_bigram[k_1][k_2]) - math.log(corpus_dict[k_1])
				else:
					corpus_bigram[k_1][k_2] = (-9999.0)
			elif smoothing == 1:
				corpus_bigram[k_1][k_2] = math.log(corpus_bigram[k_1][k_2]+K_FACTOR) - math.log((corpus_dict[k_1] + K_FACTOR*len(corpus_dict)))
			elif smoothing == 2 :
				if corpus_bigram[k_1][k_2] < GOOD_TURING_FACTOR:
					corpus_bigram[k_1][k_2] = math.log((corpus_bigram[k_1][k_2]+1)*freq_counter[corpus_bigram[k_1][k_2]+1]) - math.log(freq_counter[corpus_bigram[k_1][k_2]])
				else: 
					corpus_bigram[k_1][k_2] = math.log(corpus_bigram[k_1][k_2]) - math.log(corpus_dict[k_1])
			elif smoothing == 3 :
				if corpus_bigram[k_1][k_2] > 1: 
					corpus_bigram[k_1][k_2] = math.log(corpus_bigram[k_1][k_2] - 0.75) - math.log(corpus_dict[k_1])
				elif corpus_bigram[k_1][k_2] == 1: 
					corpus_bigram[k_1][k_2] = math.log(corpus_bigram[k_1][k_2] - 0.5) - math.log(corpus_dict[k_1])
				else:
					corpus_bigram[k_1][k_2] = 0.000027
					
	return corpus_bigram
'''
This method generates a random sentence using the Unigram model
@param n: the sentence is capped at n words
@param prob_dict: the probability distribution of the unigram model
@return a generated sentence
'''
def generative_model_uni(n, prob_dict):
	arr = np.random.choice(prob_dict.keys(), n, p = prob_dict.values())
	unigram_sentence = ' '.join(arr)
	period_index = unigram_sentence.find('.')
	if period_index != -1: # can we remove this condition?
		unigram_sentence = unigram_sentence[:period_index+1]	
	return unigram_sentence

'''
This method generates a random sentence using the Bigram model
@param n: the sentence is capped at n words
@param prob_dict_uni: the probability distribution of the unigram model
@param prob_dict_bi: the probability distribution of the bigram model
@return a generated sentence
'''
def generative_model_bi(n, prob_dict_uni, prob_dict_bi):
	arr = []
	arr.append("<s>")
	# first = np.random.choice(prob_dict_uni.keys(), 1, p= prob_dict_uni.values())[0]
	# arr.append(first)
	curr_word = "<s>"	
	i = 0
	# for i in range(n):
	# 	next_word = np.random.choice(prob_dict_bi[curr_word].keys(), 1, prob_dict_bi[curr_word].values())[0]
	# 	curr_word = next_word
	# 	arr.append(next_word)
	while (i < n):
		if curr_word in prob_dict_bi:
			#print(sum(prob_dict_bi[curr_word].values()))
			next_word = np.random.choice(prob_dict_bi[curr_word].keys(), 1, p = prob_dict_bi[curr_word].values())[0]			
		else:
			next_word = np.random.choice(prob_dict_uni.keys(), 1, p = prob_dict_uni.values())[0]

		curr_word = next_word
		arr.append(next_word)
		i += 1

	bigram_sentence = ' '.join(arr)
	period_index = bigram_sentence.find('.')
	if period_index != -1:
		bigram_sentence = bigram_sentence[:period_index+1]
	return bigram_sentence

'''
Compute pp using a model
'''
def compute_perplexity(corpus, model):
	corpus = open(corpus,'r')
	corpus_list = corpus.read().replace("\n", " ").split(" ")

	for i in range(len(corpus_list)):
		if corpus_list[i] not in model:
			corpus_list[i] = '<UNK>'

	perplexity = 0.0
	miss_count = 0
	unseen_count = 0
	for i in range(len(corpus_list)-1):
		if (corpus_list[i] in model) and (corpus_list[i+1] in model[corpus_list[i]]):
			perplexity += model[corpus_list[i]][corpus_list[i+1]]
		elif corpus_list[i] in model:
			unseen_count += 1
		else:
			miss_count += 1
	perplexity *= -1.0
	perplexity = math.exp((1.0/len(corpus_list))*perplexity)
	# print("unseen:\t" + str(unseen_count) + "\tmiss:\t" + str(miss_count))
	return perplexity

def classifier(corpus_list, model):

	for i in range(len(corpus_list)):
		if corpus_list[i] not in model:
			corpus_list[i] = '<UNK>'

	perplexity = 0.0
	miss_count = 0
	unseen_count = 0
	for i in range(len(corpus_list)-1):
		if (corpus_list[i] in model) and (corpus_list[i+1] in model[corpus_list[i]]):
			perplexity += model[corpus_list[i]][corpus_list[i+1]]
		elif corpus_list[i] in model:
			unseen_count += 1
		else:
			miss_count += 1
	perplexity *= -1.0
	perplexity = math.exp((1.0/len(corpus_list))*perplexity)
	# print("unseen:\t" + str(unseen_count) + "\tmiss:\t" + str(miss_count))
	return perplexity

def dict_exclude_keys(d, ex_keys):
	return {x:d[x] for x in d if x not in ex_keys}

# d = {'a':1, 'b':2, 'c':3}
# print(dict_exclude_keys(d, ['a','b']))









