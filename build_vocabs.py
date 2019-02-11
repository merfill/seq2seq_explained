
import nltk
import csv
import pandas as pd


def load_vocabs(data_file):
    s = set()
    t = set()
    print 'start read vocabulary from {} file...'.format(data_file)
    df = pd.read_csv(data_file, sep=',', encoding='utf8')
    for _, row in df.iterrows():
        s.update([c for c in row['before']])
        t.update([w for w in nltk.word_tokenize(row['after'])])
    return (s, t)

s = set()
t = set()
for f in ['train', 'dev', 'test']:
    (s1, t1) = load_vocabs('./data/{}.csv'.format(f))
    s = s.union(s1)
    t = t.union(t1)


def write_vocab(v, vocab_file):
    print 'write {} of elements to file {}'.format(len(v), vocab_file)
    with open(vocab_file, 'wb') as f:
        for r in sorted(list(v)):
            f.write('{}\n'.format(r.encode('utf8')))

write_vocab(s, './data/vocab.source.txt')
write_vocab(t, './data/vocab.target.txt')

