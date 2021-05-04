import sys, re, string, nltk
from math import log, sqrt
from stop_list import closed_class_stop_words as stop

#   This program is an Ad Hoc Information Retrieval task using TF-IDF 
#   weights and cosine similarity scores. Vectors are based on all
#   the words in the query sans the stop words specified. 
#   Author: Reema Amhaz 

input = open(sys.argv[1])
queries = open("cran.qry").readlines()
output = open("output.txt", "w")

#clean up the queries so they don't include the list of stop words or punctuation/digits
refined_list = {}
query_vector = {}
def process_queries(query):
    index = 0
    for q in query:
        #creates its own indexing mechanism 
        if q.split()[0] == ".I":
            index += 1
        parse = True
        #parse the query only
        if q.startswith(".I") or q.startswith(".W"):
            parse = False
        if parse:
            #split into separate word lists
            line = nltk.word_tokenize(q)
            # for word in the list 
            for w in line:
                #check if it's ok
                if w not in stop and w not in string.punctuation and not w.isdigit():
                    # strip and lower it 
                    w = w.strip("()/.").lower()
                    # list with just frequency counts in the whole doc
                    if w in refined_list:
                        refined_list[w] += 1
                    else:
                        refined_list[w] = 1
                    # keep count of each word freq for each query 
                    try:
                        try:
                            query_vector[index][w] += 1 
                        except: 
                            query_vector[index][w] = 1
                    except:
                        query_vector[index] = {w:1}
    return query_vector

abstract_vector = {}
abstract_words = {}
#clean up the abstracts so they don't include the list of stop words or punctuation/digits
def process_abstract(abstracts):
    indexed = 0
    parsing = False
    for a in abstracts:
        #creates its own indexing mechanism 
        if a.split()[0] == ".I":
            indexed += 1
        #parse the abstract only
        if a.startswith(".W"):
            parsing = True
        if parsing:
            #split into separate word lists
            lines = nltk.word_tokenize(a)
            # for word in the list 
            for l in lines:
                #check if it's ok
                if l not in stop and l not in string.punctuation and not l == ".I" and not l == ".W" and not l.isdigit():
                    # strip and lower it
                    l = l.strip("()/.").lower()
                    # list with just frequency counts in the whole doc
                    if l in abstract_words:
                        abstract_words[l] += 1
                    else:
                        abstract_words[l] = 1
                    # keep count of each word freq for each abstract
                    try:
                        try:
                            abstract_vector[indexed][l] += 1 
                        except: 
                            abstract_vector[indexed][l] = 1
                    except:
                        abstract_vector[indexed] = {l:1}
        if a.startswith(".I"):
            parsing = False
    return abstract_vector


#find the cosine similarity by finding the dot product and sqrt of the squares of 2 vectors
def cos(v1,v2):
    numerator = 0
    d1 = 0
    d2 = 0
    for i in v1:
        for j in v2:
            numerator += i*j
            d1 += i**2
            d2 += j**2
    #if the denominator is 0 we want to set the sum to 0 
    if (d1*d2 == 0):
        sum = 0
    #otherwise do the normal division
    else:
        sum = float(numerator) / sqrt(d1 * d2)
    return sum

#find the weighting using TF(t) * IDF(t), using log to slow its growth

def tfidf(vector, list):
    idf = {}
    tfidf_vector = {}
    count = len(vector.keys())
    # for each query or abstract
    for k in vector.keys():
        tfidf_vector[k] = {}
        idf[k] = {}
        # for each word in that specific query or abstract 
        for k2 in vector[k].keys():
            # get the freq in the abstract/query
            occur = list.get(k2)
            # if it's 0, this will throw an error so we just set it to 0
            if count/occur == 0:
                tfidf_vector[k][k2] = [vector[k][k2], 0, 0]
                idf[k][k2] = 0
            # each word in the query has a tf, idf, and tfidf score 
            else:
                tfidf_vector[k][k2] = [vector[k][k2], (log(count/occur)), (vector[k][k2] * (log(count/occur)))]
                idf[k][k2] = vector[k][k2] * (log(count/occur))
    return tfidf_vector, idf

process_queries(queries)
vector_A = tfidf(query_vector, refined_list)[1]
process_abstract(input)
vector_B = tfidf(abstract_vector, abstract_words)[1]

sort = {}
for q1 in vector_A:
    q_keys = vector_A[q1]
    for q2 in vector_B:
        abs_keys = vector_B[q2]
        query_matches = q_keys.values()
        abs_matches = [0] * len(query_matches)
        for k1 in abs_keys:
            if k1 in q_keys.keys():
                i = q_keys.keys().index(k1)
                abs_matches[i] = abs_keys[k1]
        if abs_matches:
            try:
                try:
                    sort[q1][q2] += cos(query_matches, abs_matches)
                except: 
                    sort[q1][q2] = cos(query_matches, abs_matches)
            except:
                sort[q1] = {q2: 0}
for query_key in sort.keys():
    s = sort[query_key]
    final = sorted(((value, key) for (key,value) in s.items()), reverse = True)
    for sets in final:
        output.write(str(query_key))
        output.write(" ")
        output.write(str(sets[1]))
        output.write(" ")
        output.write(str(sets[0]))
        output.write("\n")
