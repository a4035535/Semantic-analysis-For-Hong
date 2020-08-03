## -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from zhon.hanzi import punctuation
from zhon.hanzi import characters

SECTIONS_SPLIT_STR = '------------\n'
OTHER_PUNCTUATIONS = '.`*",!()><;[]“?'
OTHER_PAIRS = ['&qut']


def split_sections(text, save=True):
    sections = text.split(SECTIONS_SPLIT_STR)

    if save is True:
        for index, i in enumerate(sections[1:], 1):
            with open('sections/' + str(index) + '.txt', 'w', encoding='utf-8') as f:
                f.write(i)
    return sections


def delete_sections_split_str(text):
    return text.replace('------------\n', '')


def delete_multiple_spaces(text, replace=''):
    import re
    return re.sub(' +', replace, text)


def delete_punctuations(text, replace=''):
    import re
    text = re.sub(r"[%s]+" % punctuation, replace, text)
    for i in OTHER_PUNCTUATIONS:
        text = text.replace(i, '')
    for i in OTHER_PAIRS:
        text = text.replace(i, '')
    return text


def delete_null_break_line(text):
    lines = text.split('\n')
    to_del = []
    for i in range(len(lines)):
        if lines[i] == '':
            to_del.append(lines[i])
    for i in to_del:
        lines.remove(i)
    return '\n'.join(lines)


def delete_break_line(text):
    return text.replace('\n', '')


def jieba_split(text, re_type='generator'):
    import jieba
    seg_generator = jieba.cut(text)
    if re_type is 'generator':
        return seg_generator
    elif re_type is 'list':
        return list(seg_generator)


def N_gram(words_list, N=2, save=False, save_name='model'):
    grams = []
    for i in range(len(words_list) - N):
        grams.append(''.join([_ for _ in words_list[i:i + N]]))

    data = None
    if save:
        data = gram_frequency(grams)
        data.to_csv('model/model_' + save_name + '.csv', encoding='utf-8')

    return grams, data


def gram_frequency(grams):
    dir = {}
    for i in grams:
        if i in dir.keys():
            dir[i] += 1
        else:
            dir[i] = 1
    grams_list = []
    value = []
    for i, j in dir.items():
        grams_list.append(i)
        value.append(j)
    data = pd.DataFrame(zip(grams_list, value), columns=['gram', 'value'])
    return data.sort_values(by=['gram'])


def full_deal_text():
    with open('data/Stone.txt', encoding='utf-8') as f:
        text = f.read()
    text = delete_multiple_spaces(text)
    text = delete_sections_split_str(text)
    text = delete_punctuations(text)
    text = delete_break_line(text)

    with open('data/full_deal_data.txt', 'w', encoding='utf-8') as f:
        f.write(text)

    splits = jieba_split(text, 'list')
    with open('data/splited_words.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(splits))


def make_tf_idf_model(ngram_range=(1, 1), save_name='model/tf_idf.csv', show=True, mode='word'):
    sections = []
    for i in range(1, 121):
        with open('sections/' + str(i) + '.txt', encoding='utf-8') as f:
            text = f.read()
            if mode is 'word':
                words = jieba_split(text)
            elif mode is 'char':
                words = text
            sections.append(words)

    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=[' '], token_pattern='[%s]' % characters)
    X = vectorizer.fit_transform([' '.join(i) for i in sections])

    if show:
        print(X.toarray().shape)  # n=1 (120, 44110)
        print(vectorizer.get_feature_names())

    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(X)

    data = pd.DataFrame(tf_idf.toarray(), columns=vectorizer.get_feature_names())
    data.to_csv(save_name)


def train_and_show(data, cluster, pca=None):
    if pca:
        data = pca.fit_transform(data)
        data = pd.DataFrame(data)
    result = cluster.fit(data)

    front = sum(result.labels_[:80])
    back = sum(result.labels_[80:])
    print('cluster : {}'.format(cluster))
    print('front positive example : {}'.format(front / 80))
    print('back positive example : {}'.format(back / 40))


if __name__ == '__main__':
    # test about sklearn using
    ''' 
    corpus = [
        'This is the first document. This document is the second document.',
    ]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    
    print(vectorizer.get_feature_names())
    print(X.toarray())
    
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(X)
    
    X = vectorizer.fit_transform(['This is a test document'])
    print(vectorizer.get_feature_names())
    print(X.toarray())
    '''
