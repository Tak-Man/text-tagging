grammar = r"""
    NBAR:
        {<NN.*|JJS>*<NN.*>}

    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}
"""


def tokenize_chunk_parse(line):
    chunker = nltk.RegexpParser(grammar)

    toks = nltk.regexp_tokenize(line, TOKEN_RE)
    postoks = nltk.tag.pos_tag(toks)

    tree = chunker.parse(postoks)

    return [term for term in leaves(tree)]


def leaves(tree):
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        yield subtree.leaves()



import nltk
lines = 'lines is some string of words'
tokenized = nltk.word_tokenize(lines)
nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if(pos[:2] == 'NN')]
print (nouns)


from skmultilearn.model_selection import iterative_train_test_split
X_train, y_train, X_test, y_test = iterative_train_test_split(x, y, test_size = 0.1)



# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/

# Binary relevance - correlations between classes could be lost
# Classifier chains preserves label correlation
# Label powerset transforms a multi-label problem into a multi-class problem


# https://stackoverflow.com/questions/37037450/multi-label-feature-selection-using-sklearn
from sklearn.feature_selection import chi2, SelectKBest

selected_features = []
for label in labels:
    selector = SelectKBest(chi2, k='all')
    selector.fit(X, Y[label])
    selected_features.append(list(selector.scores_))

// MeanCS
selected_features = np.mean(selected_features, axis=0) > threshold
// MaxCS
selected_features = np.max(selected_features, axis=0) > threshold


