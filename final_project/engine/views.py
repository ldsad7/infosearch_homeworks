# import libraries
import csv
import json
import os
from string import punctuation

from django.shortcuts import render
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

punctuation += '…—'

import sys
from pymorphy2 import MorphAnalyzer
from collections import Counter, defaultdict
import math
import numpy as np
import pandas as pd
from scipy import spatial
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
import tensorflow as tf_
import warnings

warnings.filterwarnings('ignore')

# ~ global variables
pymorphy2_analyzer = MorphAnalyzer()
rus_stopwords = stopwords.words('russian')
path_to_tf_idf_inverted_collection = os.path.join('files', 'tf_idf_inverted_collection.json')
path_to_tf_idf_df = os.path.join('files', 'tf_idf_df.pkl')
path_to_fasttext_vectors = os.path.join('files', 'fasttext_vectors')
store = pd.HDFStore('store.h5')
use_from_dataset = 1000
top = 10
fasttext_model_file = os.path.join(os.getcwd(), '181', 'model.model')
path_to_elmo_vectors = os.path.join('files', 'elmo_vectors.npy')


############## Read file

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        # skip the header
        print('header:', next(csv_reader, None))
        queries = []
        docs = []
        answers = []
        for _ in csv_reader:
            for _, query, doc, answer in csv_reader:
                queries.append(query)
                docs.append(doc)
                answers.append(int(float(answer)))
        return queries, docs, answers


##############

_, docs, _ = read_file('quora_question_pairs_rus.csv')
docs = docs[:use_from_dataset]


############## Common functions

def preprocess_text(text, fast=True):
    lowered_tokens = [
        word.strip(punctuation) for word in word_tokenize(text.lower()) if word.strip(punctuation)
    ]
    if fast:
        return lowered_tokens
    return [pymorphy2_analyzer.normal_forms(token)[0] for token in lowered_tokens]


def get_most_probable_docs(doc_vector, docs_matrix):
    cosine_values = cosine_similarity(
        docs_matrix, doc_vector.reshape(1, -1)).reshape(docs_matrix.shape[0])
    return [(docs[doc_id], cosine_value) for doc_id, cosine_value in sorted(list(
        enumerate(cosine_values)), key=lambda elem: elem[1], reverse=True)[:top]]


############## Elmo

def load_elmo_embeddings(directory, top=False):
    """
    :param directory: directory with an ELMo model ('model.hdf5', 'options.json' and 'vocab.txt.gz')
    :param top: use ony top ELMo layer
    :return: ELMo batcher, character id placeholders, op object
    """
    vocab_file = os.path.join(directory, 'vocab.txt')
    options_file = os.path.join(directory, 'options.json')
    weight_file = os.path.join(directory, 'model.hdf5')

    # Create a Batcher to map text to character ids.
    batcher = Batcher(vocab_file, 50)

    # Input placeholders to the biLM.
    sentence_character_ids = tf_.placeholder('int32', shape=(None, None, 50))

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(options_file, weight_file, max_batch_size=300)

    # Get ops to compute the LM embeddings.
    sentence_embeddings_op = bilm(sentence_character_ids)

    # Get an op to compute ELMo (weighted average of the internal biLM layers)
    elmo_sentence_input = weight_layers('input', sentence_embeddings_op, use_top_only=top)
    return batcher, sentence_character_ids, elmo_sentence_input


def get_elmo_vectors(sess, texts, batcher, sentence_character_ids, elmo_sentence_input):
    """
    :param sess: TensorFlow session
    :param texts: list of sentences (lists of words)
    :param batcher: ELMo batcher object
    :param sentence_character_ids: ELMo character id placeholders
    :param elmo_sentence_input: ELMo op object
    :return: embedding matrix for all sentences (max word count by vector size)
    """

    # Create batches of data.
    sentence_ids = batcher.batch_sentences(texts)
    print('Sentences in this batch:', len(texts), file=sys.stderr)

    # Compute ELMo representations.
    elmo_sentence_input_ = sess.run(elmo_sentence_input['weighted_op'],
                                    feed_dict={sentence_character_ids: sentence_ids})

    return elmo_sentence_input_

elmo_tokenized_docs = [preprocess_text(doc) for doc in docs]
elmo_vector_size = 1024

def search_elmo(query):
    batcher, sentence_character_ids, elmo_sentence_input = load_elmo_embeddings(os.path.join(os.getcwd(), '196'))
    tokenized_query = preprocess_text(query)

    elmo_vectors = np.zeros((len(elmo_tokenized_docs), elmo_vector_size))

    with tf_.variable_scope("other_charge", reuse=True) as scope:
        with tf_.Session() as sess:
            sess.run(tf_.global_variables_initializer())
            elmo_vector = np.mean(get_elmo_vectors(
                sess, [tokenized_query], batcher, sentence_character_ids, elmo_sentence_input), axis=1)[0]

            if os.path.exists(path_to_elmo_vectors):
                elmo_vectors = np.load(path_to_elmo_vectors)
            else:
                batch_size = 300
                for i in range(0, len(elmo_tokenized_docs), batch_size):
                    elmo_vectors[i:i + batch_size] = np.mean(get_elmo_vectors(
                        sess, elmo_tokenized_docs[i:i + batch_size], batcher, sentence_character_ids, elmo_sentence_input),
                        axis=1)
                np.save(path_to_elmo_vectors, elmo_vectors)

    return get_most_probable_docs(elmo_vector, elmo_vectors)


##############


############## TF-IDF

def inverted_index(text, text_id):
    text_length = len(text)
    return {key: (text_id, value, text_length) for key, value in Counter(text).items()}


def get_inverted_collection(docs):
    inverted_collection = defaultdict(list)
    for doc_num, doc in enumerate(docs):
        print(doc_num)
        for token, value in inverted_index(preprocess_text(doc), doc_num).items():
            inverted_collection[token].append(value)
    return inverted_collection


def tf_idf_search(query, tf_idf_df, tf_idf_inverted_collection):
    query_dct = inverted_index(preprocess_text(query), 'query')
    removed_keys = set(query_dct.keys()) - set(tf_idf_inverted_collection.keys())
    for removed_key in removed_keys:
        del query_dct[removed_key]

    query_vec = np.zeros((1, tf_idf_vec_size))
    columns_list = tf_idf_columns.values.tolist()
    for word, triple in query_dct.items():
        tf = triple[1] / triple[2]
        idf = math.log10(tf_idf_num_of_docs / len(tf_idf_inverted_collection[word]))
        query_vec[0][columns_list.index(word)] = tf * idf

    rows_list = tf_idf_index.values.tolist()
    doc_to_cosine = {}
    for row in rows_list:
        doc_to_cosine[row] = 1 - spatial.distance.cosine(tf_idf_df.loc[row], query_vec)

    return [(docs[pair[0]], pair[1]) for pair in sorted(doc_to_cosine.items(), key=lambda elem: -elem[1])[:top]]


if os.path.isfile(path_to_tf_idf_inverted_collection):
    with open(path_to_tf_idf_inverted_collection, 'r', encoding='utf-8') as f:
        tf_idf_inverted_collection = json.load(f)
else:
    tf_idf_inverted_collection = get_inverted_collection(docs)
    with open(path_to_tf_idf_inverted_collection, 'w', encoding='utf-8') as f:
        json.dump(tf_idf_inverted_collection, f)

if 'tf_idf_df' in store:
    tf_idf_df = store['tf_idf_df']
else:
    vec_size = len(tf_idf_inverted_collection)
    print(vec_size, 'words')
    unique_docs = {triple[0] for triples in tf_idf_inverted_collection.values() for triple in triples}
    num_of_docs = len(unique_docs)
    print(num_of_docs, 'docs')

    index = pd.Index(unique_docs, name='docs')
    columns = pd.Index(tf_idf_inverted_collection.keys(), name='words')
    tf_idf_df = pd.DataFrame(0., index=index, columns=columns)
    for word, triples in tf_idf_inverted_collection.items():
        # I didn't add 1 to the denominator as pairs is definitely not an empty array
        idf = math.log10(num_of_docs / len(triples))
        for triple in triples:
            tf = triple[1] / triple[2]
            tf_idf_df.at[triple[0], word] = tf * idf
    store['tf_idf_df'] = tf_idf_df

tf_idf_vec_size = len(tf_idf_inverted_collection)
tf_idf_unique_docs = {triple[0] for triples in tf_idf_inverted_collection.values() for triple in triples}
tf_idf_num_of_docs = len(tf_idf_unique_docs)
tf_idf_index = pd.Index(tf_idf_unique_docs, name='docs')
tf_idf_columns = pd.Index(tf_idf_inverted_collection.keys(), name='words')


def search_tf_idf(query):
    return tf_idf_search(query, tf_idf_df, tf_idf_inverted_collection)


############## BM-25

bm25_N = len(docs)
bm25_doc_lens = np.array([len(doc.split()) for doc in docs])
bm25_average_doc_length = sum(bm25_doc_lens) / bm25_N
bm25_vect = TfidfVectorizer(use_idf=False)

if 'bm25_df' in store:
    bm25_df = store['bm25_df']
else:
    bm25_tf_matrix = bm25_vect.fit_transform(docs)
    bm25_df = pd.DataFrame(bm25_tf_matrix.toarray(), columns=bm25_vect.get_feature_names())
    store['bm25_df'] = bm25_df


def bm25_search(query, k=2.0, b=0.75):
    query_vect = TfidfVectorizer(use_idf=False)
    query_vect.fit_transform([query])
    features = query_vect.get_feature_names()
    df = bm25_df[features]
    df = df / (df + k * np.repeat((1 - b + b * bm25_doc_lens.reshape((bm25_N, 1)) / bm25_average_doc_length),
                                  len(features), axis=1))
    nq = df.astype(bool).sum(axis=0)
    idfs = np.log((bm25_N - nq + 0.5) / (nq + 0.5))
    scores = df.dot(idfs * (k + 1))
    return [(docs[index], scores[index]) for index in np.argsort(scores)[::-1][:top]]


def search_bm25(query):
    return bm25_search(query)


##############


############## Fasttext

def get_fasttext_vector(tokenized_doc, fasttext_model):
    # создаем маски для векторов
    lemmas_vectors = np.zeros((len(tokenized_doc), fasttext_model.vector_size))
    vec = np.zeros((fasttext_model.vector_size,))

    # если слово есть в модели, берем его вектор
    for idx, lemma in enumerate(tokenized_doc):
        if lemma in fasttext_model.vocab:
            lemmas_vectors[idx] = fasttext_model.wv[lemma]

    return np.mean(lemmas_vectors, axis=0)


fasttext_model = KeyedVectors.load(fasttext_model_file)

if os.path.exists(path_to_fasttext_vectors):
    fasttext_vectors = store['fasttext_vectors']
else:
    fasttext_tokenized_docs = [preprocess_text(doc) for doc in docs]
    fasttext_vectors = np.zeros((len(fasttext_tokenized_docs), fasttext_model.vector_size))
    for doc_index, tokenized_doc in enumerate(fasttext_tokenized_docs):
        fasttext_vectors[doc_index] = get_fasttext_vector(tokenized_doc, fasttext_model)
    store['fasttext_vectors'] = pd.DataFrame(fasttext_vectors)


def fasttext_search(query):
    return get_most_probable_docs(
        get_fasttext_vector(preprocess_text(query), fasttext_model), fasttext_vectors
    )


def search_fasttext(query):
    return fasttext_search(query)


def main(request):
    query = request.GET.get('query')
    engine = request.GET.get('engine')
    context = {'docs': [], 'engine': ''}
    if query and engine:
        context['engine'] = engine
        if engine == 'tf-idf':
            context['docs'] = search_tf_idf(query)
        elif engine == 'bm25':
            context['docs'] = search_bm25(query)
        elif engine == 'fasttext':
            context['docs'] = search_fasttext(query)
        elif engine == 'elmo':
            context['docs'] = search_elmo(query)

    print(context)
    return render(request, 'main.html', context)
