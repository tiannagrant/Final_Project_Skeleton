import pandas as pd
import numpy as np
from Bio import SeqIO
from difflib import SequenceMatcher
import re
import random

pos_seq_file = "/Users/tiannagrant/Downloads/Final_Project_Skeleton-master/data/rap1-lieb-positives.txt"
neg_seq_file = "/Users/tiannagrant/Downloads/Final_Project_Skeleton-master/data/yeast-upstream-1k-negative.fa"
test_seq_file = "/Users/tiannagrant/Downloads/Final_Project_Skeleton-master/data/rap1-lieb-test.txt"

def read_seqs(pos_seq_file, neg_seq_file, test_seq_file):
    #Reading in the training and testing sequences using open
    #
    pos_seq = open(pos_seq_file).read().splitlines()
    test_seq = open(test_seq_file).read().splitlines()
    if '.fa' in neg_seq_file:
        neg_sequences = [str(fasta.seq) for fasta in SeqIO.parse(open(neg_seq_file),'fasta')]
    elif ".txt" in neg_seq_file:
        neg_sequences = open(neg_seq_file).read().splitlines()


    return pos_seq, neg_sequences, test_seq

    #I decided to remove negative sequences that have postive sub-sequences
    #Input is negative sequences and postive sequences, setting a string length of pos_seq

def trim_neg_sequences(pos_seqs, neg_seqs, bp=17, ratio = 4):

        #Drop negative sequences that are a sub-sequence of the positive sequences
        #INPUT: positive sequences and negative sequences, length of bases to keep, and ratio for
        #number of negative sequences to positive sequences
        #OUTPUT: negative sequences that are not similar to positive sequences and
        #are the same number of base pairs as positive sequences

        for pos in pos_seqs:
            for neg in neg_seqs:
                if re.search(pos, neg):
                    neg_seqs.remove(neg)
            # downsampling of negative sequences
        neg_keep = int(len(pos_seqs)*ratio)
        neg_seqs_sub = np.random.choice(neg_seqs, size=neg_keep, replace=False)

        short_neg = []
        for neg in neg_seqs_sub:
            rand_start = np.random.randint(0, len(neg)-bp+1)
            short_neg.append(neg[rand_start:rand_start+bp])
        return short_neg


def one_hot_neg(trim):
            one_hot_neg = []
            for neg in trim:
                one_hot_neg.append(encode_DNA(neg).reshape(68,1))
            return one_hot_neg

        #One hot positive

def one_hot_pos(pos_seq):
            one_hot_pos = []
            for pos in pos_seq:
                #Calling Encode_DNA function
                one_hot_pos.append(encode_DNA(pos).reshape(68,1))
            return one_hot_pos



def encode_DNA(seq):
                """
                Convert DNA sequence to binary values for input into neural net
                INPUT: DNA sequence
                OUTPUT: binary encoding of sequence
                """
                seq2bin_dict = {'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'T':[0,0,0,1]}
                return np.array(sum([seq2bin_dict.get(nuc) for nuc in seq], []))


def split_data(pos_seqs, neg_seqs, split= 0.5):
                    """
                    Split data with known outcomes into training and testing data
                    INPUT: positive examples, negative examples, and percent to keep for training
                    OUTPUT: positive and negative training sets and
                    positive and negative testing sets
                    """

                    pos_size, neg_size = int(len(pos_seqs)*split), int(len(neg_seqs)*split)
                    train_pos = np.random.choice(range(0,pos_size), size=pos_size , replace=False)
                    print('train.pos',train_pos.shape)
                    train_neg = np.random.choice(range(0,neg_size), size=pos_size,  replace=False)
                    print(pos_seqs[train_pos].shape)

                    test_pos =  set(range(0,len(pos_seqs)))- set(train_pos)

                    #below is the original
                    #test_pos = set(pos_seqs[68,1])-set(pos_seqs[train_pos[0,68]])
                    #test_neg = set(neg_seqs)-set(train_seqs[train_neg])
                    test_neg = set(range(0,len(neg_seqs)))- set(train_neg)
                    return pos_seqs[train_pos], neg_seqs[train_neg]

def combine_and_shuffle(pos, neg):
    combined = np.concatenate((pos, neg))
    expected = np.append(np.array([[1]]*len(pos)), np.array([[0]]*len(neg)))
    shuf_combined, shuf_expected = random.shuffle(combined, expected)
    return shuf_combined, np.reshape(shuf_expected, (len(shuf_expected),1))

def preprocess(pos_seqs, neg_seqs, split=0.8):
	"""
	process, split, and encode data
	INPUT: positive and negative sequences
	OUTPUT: training and testing encoded sequences
	"""
	# split the data into training and testing sets
	train_p_l, train_n_l, test_p_l, test_n_l = split_data(pos_seqs, neg_seqs, split)
	train_p_b = np.array([encode_DNA(seq) for seq in train_p_l])
	train_n_b = np.array([encode_DNA(seq) for seq in train_n_l])

	# encode dna from nucleotides to binary output
	test_p_b = np.array([encode_DNA(seq) for seq in test_p_l])
	test_n_b = np.array([encode_DNA(seq) for seq in test_n_l])

	# Combine training positive and negative sequences
	train_seq, train_exp = combine_and_shuffle(train_p_b, train_n_b)
	if split != 1:
		test_seq, test_exp = combine_and_shuffle(test_p_b, test_n_b)
	else:
		test_seq, test_exp = [],[]

	return train_seq, train_exp, test_seq, test_exp

def shuffle(sequences, phenotype):
	"""
	Shuffle inputs and outputs
	"""

	seqs = pd.DataFrame(sequences)
	phen = pd.DataFrame(phenotype)
	df = pd.concat([seqs,phen], axis=1)
	df_shuf = df.sample(frac=1)
	shuf_seqs = np.array(df_shuf.iloc[:,:-1])
	shuf_phen = np.array(df_shuf.iloc[:,-1])
	return shuf_seqs, shuf_phen
