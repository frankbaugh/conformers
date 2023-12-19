import datautils, pickle

path = '/Users/frank/code/conf/37Conf8/'
anidict = datautils.get_anidict(path)

with open('anidict.pickle', 'wb') as handle:
    pickle.dump(anidict, handle)