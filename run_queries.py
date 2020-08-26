import argparse
import os

from inverted_index import InvertedIndex
from preprocessor import Preprocessor
from similarity_measures import TF_Similarity, TFIDF_Similarity, BM25_Similarity
import string

parser = argparse.ArgumentParser(description='Run all queries on the inverted index.')
parser.add_argument('--new', default=True, help='If True then build a new index from scratch. If False then attempt to'
                                                ' reuse existing index')
parser.add_argument('--sim', default='TF', help='The type of similarity to use. Should be "TF" or "TFIDF"')
args = parser.parse_args()

index = InvertedIndex(Preprocessor())
index.index_directory(os.path.join('gov', 'documents'), use_stored_index=(not args.new))

sim_name_to_class = {'TF': TF_Similarity,
                     'TFIDF': TFIDF_Similarity,
                     'BM25': BM25_Similarity}

sim = sim_name_to_class[args.sim]
index.set_similarity(sim)
print(f'Setting similarity to {sim.__name__}')

print()
print('Index ready.')


topics_file = os.path.join('gov', 'topics', 'gov.topics')
runs_file = os.path.join('runs', 'retrieved.runs')

# TODO run queries
"""
You will need to:
    1. Read in the topics_file.
    2. For each line in the topics file create a query string (note each line has both a query_id and query_text,
       you just want to search for the text)  and run this query on index with index.run_query().
    3. Write the results of the query to runs_file IN TREC_EVAL FORMAT
        - Trec eval format requires that each retrieval is on a separate line of the form
          query_id Q0 document_id rank similarity_score MY_IR_SYSTEM
"""
f = open(topics_file, "r")
line = f.readline()
write_file = open(runs_file, 'w')
while line:
    del_estr = string.punctuation
    replace = " " * len(del_estr)
    tran_tab = str.maketrans(del_estr, replace)
    line = line.translate(tran_tab)
    terms = line.split(' ')
    Q_index = terms[0]
    query = ""
    for t in terms[1:]:
        query = query+" "+t
    sim_scores = index.run_query(query)
    for i in range(len(sim_scores)):
        this_string = Q_index+" "+"Q0"+" "+sim_scores[i][0]+" "+str(i)+" "+str(sim_scores[i][1])+" "+"MY_IR_SYSTEM"+"\n"
        write_file.write(this_string)
    line = f.readline()
write_file.close()
f.close()
