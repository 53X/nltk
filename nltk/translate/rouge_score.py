'''ROUGE score implementation
Link to paper:
http://www.aclweb.org/anthology/W04-1013
'''

from nltk.util import ngrams, skipgrams
from util import jacknifing, rouge_lcs
import numpy as np


def get_score(r_lcs, p_lcs, beta=1):
    '''
    This is a scoring function implementing the
    F-measure score used in the paper.
    Here:
    
    :param beta : parameter
    :type beta : float
    
    :param r_lcs : recall factor
    :type r_lcs : float
    
    :param p_lcs : precision factor
    :type p_lcs : float

    
    According to the paper the formula for the
    F-measure score goes as :

    
    (1+beta**2)*r_lcs*p_lcs)/(r_lcs+(beta**2)*p_lcs)

    
    '''
    try:
        return ((1+beta**2)*r_lcs*p_lcs)/(r_lcs+(beta**2)*p_lcs)
    except ZeroDivisionError as e:
        return 0


def rouge_n(references, candidate, n, averaging=True):
    ''' It is a n-gram recall between a candidate summary
    and a set of reference summaries.
        
    :param references : list of references. Every reference
    				   should be represented as a list of
    				   its tokens
    :type references : list(list(str))

    :param candidate :  the tokenized candidate string
    :type candidate : list(str)
    
    :param n : length of ngram
    :type n : int

    :param averaging : Jacknifing occurs if averaging is True
    :type averaging : Boolean
    
    ngram_cand : generator of ngrams in candidate
    
    ngram_ref : list of ngrams in reference
    
    rouge_recall : list containing all the rouge-n scores for
                   every reference against the candidate

    

    
    
    For a given candidate sentence and a reference sentence the 
    ROUGE-N score is calculated as follows:

    Let,

    matches = no. of matching n_grams btw the candidate and the reference
    total_ngram = total no. of ngrams in the reference 

    Then,

    ROUGE_N = matches/total_ngram

    If multiple references are present , then Jacknifing procedure is 
    used.

    


    >>> reference = 'police killed the gunman'
    >>> candidate_1 = 'police kill the gunman'
    >>> candidate_2 = 'the gunman kill police'
    >>> ref_list = [reference.split(' ')]


    >>> round(rouge_n(ref_list, candidate_1.split(' '), n=2), 2)
    0.33

    >>> round(rouge_n(ref_list, candidate_2.split(' '), n=2), 2)
    0.33


    '''
    ngram_cand = ngrams(candidate, n)
    rouge_recall = []
    for ref in references:
        matches = 0  #variable counting the no.of matching ngrams
        ngram_ref = list(ngrams(ref, n))
        for ngr in ngram_cand:
            if ngr in ngram_ref:
                matches += 1 
        rouge_recall.append(matches/len(ngram_ref))
    return jacknifing(rouge_recall, averaging=averaging)






def sentence_rouge_l(references, candidate, beta=1, averaging=True):
    ''' It calculates the rouge-l score between the candidate
    and the reference at the sentence level.
    
    :param references : list of reference sentences. Every reference
    				   sentence should be represented as a list of its 
    				   tokens
    :type references : list(list(str))
    
    :param candidate : tokenized candidate sentence
    :type candidate : list(str)
    
    :param beta : user-defined parameter
    :type beta : float
    
    score_list : list containing all the rouge scores for
                   every reference against the candidate
    
    r_lcs : recall factor
    
    p_lcs : precision factor
    
    score : rouge-l score between a reference sentence and 
    		the candidate sentence


    >>> reference = 'police killed the gunman'
    >>> candidate_1 = 'police kill the gunman'
    >>> candidate_2 = 'the gunman kill police'
    >>> ref_list = [reference.split(' ')]


    >>> round(sentence_rouge_l(ref_list, candidate_1.split(' ')), 2)
    0.75

    >>> round(sentence_rouge_l(ref_list, candidate_2.split(' ')), 2)
    0.5        
    
    '''
    score_list = []
    for ref in references:
        r_lcs = rouge_lcs(ref, candidate)/len(ref)
        p_lcs = rouge_lcs(ref, candidate)/len(candidate)
        score = get_score(r_lcs, p_lcs, beta=beta)
        score_list.append(score)
    return jacknifing(score_list, averaging=averaging)


def summary_rouge_l(references, candidate, beta=1, averaging=True):
    ''' It calculates the rouge-l score between the candidate
    and the reference at the summary level.
    
    param references : a corpus of lists of reference sentences, w.r.t. hypotheses
    type (references) : list(list(list(str)))
    
    param candidate :  a list of hypothesis sentences in the candidate
    type (candidate) : list(list(str))

    param beta : user-defined parameter
    type (beta) : float
    
    score_list : list containing all the rouge-l scores for
                 every reference against the candidate at the
                 summary level. 

    
    r_lcs : recall factor
    
    p_lcs : precision factor
    
    score : rouge-l score between a reference and the candidate



    >>> reference = 'police killed the gunman'
    >>> candidate_1 = 'police kill the gunman'
    >>> candidate_2 = 'the gunman kill police'
    >>> ref_list = [[reference.split(' ')]]

    >>> round(summary_rouge_l(ref_list, [candidate_1.split(' ')]), 2)
    0.75

    >>> round(summary_rouge_l(ref_list, [candidate_2.split(' ')]), 2)
    0.5
    
    '''
    score_list = []
    
    total_candidate_words = 0 # variable calculating the  total words in the candidate
    
    for cand_sent in candidate: # iterating over the word-tokenized candidate sentences

        total_candidate_words += len(cand_sent)


    for ref in references: # iterating over the list of references
                
        union_value = 0 # variable counting the length of Union LCS between 
                      # every reference sent. and candidate sent.
        
        total_reference_words = 0 # variable counting the total words in a reference 
        
        for ref_sent in ref:      # iterating over reference sentences
            
            l_ = [] # list storing the LCS words (duplicates included) for a reference 
                    # sentence and all the candidate sentences.
            
            total_reference_words += len(ref_sent)
            
            for cand_sent in candidate:  # iterating over candidate sentences
                
                d = rouge_lcs(ref_sent, cand_sent,return_string=True).strip(' ').split(' ') 
                                       
                l_ += d
             
            union_value = (union_value+len(set(l_)))/len(l_)
            
        r_lcs = union_value/total_reference_words
        p_lcs = union_value/total_candidate_words
        score = get_score(r_lcs, p_lcs, beta=beta)
        score_list.append(score)
    return jacknifing(score_list, averaging=averaging)


def normalized_pairwise_lcs(references, candidate, beta, averaging=True):
    ''' It calculates the normalized pairwaise lcs score
    between the candidate and the reference at the summary level.
    
    param references : a corpus of lists of reference sentences, w.r.t. hypotheses
    type references : list(list(list(str)))
    
    param candidate : a list of candidate sentences
    type candidate : list(list(str))
    
    param beta : parameter for the calculation of F-Score
    type beta : float

    normalized_list : list containing all the scores for
                      every reference against the candidate
    cand_sent_list = list of sentences in the candidate
    ref_sent_list = list of sentences in the reference
    arg1 = list of words in a sentence of a reference
    arg2 = list of words in a sentence of a candidate
    scr = list having the max values of lcs for every reference
          sentence when it are compared with every candidate
          sentence.
    r_lcs : recall factor
    
    p_lcs : precision factor
    
    score : desired score between a reference and the candidate
    '''
    normalized_list = []
    cand_sent_list = sentence_tokenizer.tokenize(candidate)
    for ref in references:
        ref_sent_list = sentence_tokenizer.tokenize(ref)
        scr = []
        for r_sent in ref_sent_list:
            s = []
            arg1 = tokenizer.tokenize(r_sent)
            for c_sent in candidate:
                arg2 = tokenizer.tokenize(c_sent)
                s.append(lcs(arg1, arg2, len(arg1), len(arg2))[0])
            scr.append(max(s))
        r_lcs = 2*sum(scr)/len(tokenizer.tokenize(ref))
        p_lcs = 2*sum(scr)/len(tokenizer.tokenize(candidate))
        score = get_score(r_lcs, p_lcs, beta=beta)
        normalized_list.append(score)
    return jacknifing(normalized_list, averaging=averaging)


def rouge_s(references, candidate, beta, d_skip=None,
            averaging=True, smoothing=False):
    '''
    It implements the ROUGE-S and ROUGE-SU scores.
    The skip-bigram concept has been used here.
    
    :param references : list of all references where all refernces 
                       have been tokenized into words
    :type references : list

    :param candidate : list of words in the candidate string
    :type candidate : list

    :param beta : user-defined parameter for the calculation 
                 of F1 score
    : type beta : float             
    

    :param d_skip : the distance(k) parameter for skipgram
    : type d_skip : int
    

    :param smoothing : setting this to True allows for
                       ROUGE-SU implementation.The ROUGE-SU
                       implementation helps in the unigram
                       smoothing.
    :type smoothing : boolean
                      
     
    k_c : distance parameter for candidate in the skipgram
    
    k_ref : distance parameter for reference in the skipgram
    
    cand_skip_list : list of all skipgrams of the candidate
    
    ref_skip_list : list of all skipgrams of the reference
    
    r_skip : recall factor
    
    p_skip : precision factor
    
    score : rouge-s(or SU) score between a reference and the candidate

    rouge_s_list : list of the rouge-s ( or SU) scores
                   for every reference and the candidate
    
    '''
    rouge_s_list = []
    k_c = len(candidate) if d_skip is None else d_skip
    cand_skip_list = list(skipgrams(candidate, n=2, k=k_c))
    for ref in references:
        k_ref = len(ref) if d_skip is None else d_skip
        ref_skip_list = list(skipgrams(ref, n=2, k=k_ref))
        count = 0
        for bigram in cand_skip_list:
            if bigram in ref_skip_list:
                count = count+1
        if not smoothing:
            r_skip = count/len(ref_skip_list)
            p_skip = count/len(cand_skip_list)
        else:
            for ungm in candidate:
                if ungm in ref:
                    count += 1
            r_skip = count/(len(ref_skip_list)+len(ref))
            p_skip = count/(len(cand_skip_list)+len(cand))
        score = get_score(r_skip, p_skip, beta)
        rouge_s_list.append(score)
    return jacknifing(rouge_s_list, averaging=averaging)

