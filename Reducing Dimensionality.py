# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 17:17:38 2023

@author: swank
"""

#Understanding SVD
#Singular Value Decomposition

#Essentially multiplies 3 matrices, order is:
    #Square Matrix, Diagonal Matrix, Square Matrix
    
    
#Looking for dimensionality reduction
import numpy as np
A = np.array([[1, 3, 4], [2, 3, 5], [1, 2, 3], [5, 4, 6]])
print(A)
U, s, Vh = np.linalg.svd(A, full_matrices=False)
print(np.shape(U), np.shape(s), np.shape(Vh))
print(s)
print(np.dot(np.dot(U, np.diag(s)), Vh)) # Full matrix reconstruction
print(np.round(np.dot(np.dot(U[:,:2], np.diag(s[:2])),
                      Vh[:2,:]),1)) # k=2 reconstruction
print(np.round(np.dot(np.dot(U[:,:1], np.diag(s[:1])), 
                      Vh[:1,:]),1)) # k=1 reconstruction


#Performing Factor Analysis
#Looking for hidden factors


from sklearn.datasets import load_iris
from sklearn.decomposition import FactorAnalysis

#who knows if this will load in Spyder/PyCharm -> iris is depricated Dataset

iris = load_iris()
X = iris.data
Y = iris.target
cols = [s[:12].strip() for s in iris.feature_names]
factor = FactorAnalysis(n_components=4).fit(X)
import pandas as pd
print(pd.DataFrame(factor.components_, columns=cols))

#Achieving dimensionality reduction
from sklearn.decomposition import PCA
import pandas as pd
pca = PCA().fit(X)
print('Explained variance by each component: %s' 
      % pca.explained_variance_ratio_)
print(pd.DataFrame(pca.components_, columns=cols))
Squeezing information with t-SNE
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
ground_truth = digits.target
​
#TSNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, 
            init='pca', 
            random_state=0,
            perplexity=50, 
            early_exaggeration=25,
            n_iter=300)
Tx = tsne.fit_transform(X)


import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
plt.xticks([], [])
plt.yticks([], [])
for target in np.unique(ground_truth):
    selection = ground_truth==target
    X1, X2 = Tx[selection, 0], Tx[selection, 1]
    c1, c2 = np.median(X1), np.median(X2)
    plt.plot(X1, X2, 'o', ms=5)
    plt.text(c1, c2, target, fontsize=18)

#Understanding Some Applications
#Recognizing faces with PCA
# "Sci-Kit Learn leads to PyTorch, Facial recognition with PyTorch leads to the darkside"


from sklearn.datasets import fetch_olivetti_faces
dataset = fetch_olivetti_faces(shuffle=True, 
                               random_state=101)
train_faces = dataset.data[:350,:]
test_faces  = dataset.data[350:,:]
train_answers = dataset.target[:350]
test_answers = dataset.target[350:]
print(dataset.DESCR)

####################################################################

.. _olivetti_faces_dataset:

The Olivetti faces dataset
--------------------------

`This dataset contains a set of face images`_ taken between April 1992 and 
April 1994 at AT&T Laboratories Cambridge. The
:func:`sklearn.datasets.fetch_olivetti_faces` function is the data
fetching / caching function that downloads the data
archive from AT&T.

.. _This dataset contains a set of face images: https://cam-orl.co.uk/facedatabase.html

As described on the original website:

    There are ten different images of each of 40 distinct subjects. For some
    subjects, the images were taken at different times, varying the lighting,
    facial expressions (open / closed eyes, smiling / not smiling) and facial
    details (glasses / no glasses). All the images were taken against a dark
    homogeneous background with the subjects in an upright, frontal position 
    (with tolerance for some side movement).

**Data Set Characteristics:**

    =================   =====================
    Classes                                40
    Samples total                         400
    Dimensionality                       4096
    Features            real, between 0 and 1
    =================   =====================

The image is quantized to 256 grey levels and stored as unsigned 8-bit 
integers; the loader will convert these to floating point values on the 
interval [0, 1], which are easier to work with for many algorithms.

The "target" for this database is an integer from 0 to 39 indicating the
identity of the person pictured; however, with only 10 examples per class, this
relatively small dataset is more interesting from an unsupervised or
semi-supervised perspective.

The original dataset consisted of 92 x 112, while the version available here
consists of 64x64 images.

When using these images, please give credit to AT&T Laboratories Cambridge.

####################################################################

#PCA is Principal Component Analysis

from sklearn.decomposition import RandomizedPCA
n_components = 25
Rpca = PCA(svd_solver='randomized', 
           n_components=n_components, 
           whiten=True)
Rpca.fit(train_faces)
print('Explained variance by %i components: %0.3f' 
      % (n_components, np.sum(Rpca.explained_variance_ratio_)))
compressed_train_faces = Rpca.transform(train_faces)
compressed_test_faces  = Rpca.transform(test_faces)
import matplotlib.pyplot as plt
%matplotlib inline
​
photo = 17
print('The represented person is subject %i' 
      % test_answers[photo])
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Unknown photo '+str(photo)+' in test set')
plt.imshow(test_faces[photo].reshape(64,64), 
           cmap=plt.cm.gray, interpolation='nearest')
plt.show()
mask = compressed_test_faces[photo,] 
squared_errors = np.sum((compressed_train_faces 
                         - mask)**2, axis=1)
minimum_error_face = np.argmin(squared_errors)
most_resembling = list(np.where(squared_errors < 20)[0])
print('Best resembling subject in training set: %i' 
      % train_answers[minimum_error_face])


import matplotlib.pyplot as plt
plt.subplot(2, 2, 1)
plt.axis('off')
plt.title('Unknown face '+str(photo)+' in test set')
plt.imshow(test_faces[photo].reshape(64, 64), 
           cmap=plt.cm.gray, 
           interpolation='nearest')
for k,m in enumerate(most_resembling[:3]):
    plt.subplot(2, 2, 2+k)
    plt.title('Match in train set no. '+str(m))
    plt.axis('off')
    plt.imshow(train_faces[m].reshape(64, 64), 
               cmap=plt.cm.gray, 
               interpolation='nearest')
plt.show()


#Extracting topics with NMF
#NMF is Non-Negative Matrix Factorization

from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, 
                             categories = ['misc.forsale'], 
                             remove=('headers', 'footers', 'quotes'), 
                             random_state=101)
print('Posts: %i' % len(dataset.data))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
​
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, 
                             stop_words='english')
tfidf = vectorizer.fit_transform(dataset.data)
​
n_topics = 5
nmf = NMF(n_components=n_topics, 
          random_state=101).fit(tfidf)
feature_names = vectorizer.get_feature_names()
n_top_words = 15
for topic_idx, topic in enumerate(nmf.components_):
    print('Topic #%d:' % (topic_idx+1),)
    topics = topic.argsort()[:-n_top_words - 1:-1]
    print(' '.join([feature_names[i] for i in topics]))
print(nmf.components_[0,:].argsort()[:-n_top_words-1:-1])
word_index = 2463
print(vectorizer.get_feature_names()[word_index])


#Recommending movies
#recommending system using random forest

import os
print(os.getcwd())
import pandas as pd
from scipy.sparse import csr_matrix

users = pd.read_table('ml-1m/users.dat', sep='::', 
        header=None, names=['user_id', 'gender', 
        'age', 'occupation', 'zip'], engine='python')
ratings = pd.read_table('ml-1m/ratings.dat', sep='::', 
          header=None, names=['user_id', 'movie_id', 
          'rating', 'timestamp'], engine='python')
movies = pd.read_table('ml-1m/movies.dat', sep='::', 
         header=None, names=['movie_id', 'title', 
         'genres'], engine='python')
MovieLens = pd.merge(pd.merge(ratings, users), movies)
ratings_mtx_df = MovieLens.pivot_table(values='rating', 
        index='user_id', columns='title', fill_value=0)
movie_index = ratings_mtx_df.columns
from sklearn.decomposition import TruncatedSVD
recom = TruncatedSVD(n_components=15, random_state=101)
R = recom.fit_transform(ratings_mtx_df.values.T)

movie = 'Star Wars: Episode V \
- The Empire Strikes Back (1980)'
movie_idx = list(movie_index).index(movie)
print("movie index: %i" %movie_idx)
print(R[movie_idx])


#import numpy as np
#too many imports, Just gonna leave them, it doesn'thurt anything I don't think. just git pull to ipynb.

correlation_matrix = np.corrcoef(R)
P = correlation_matrix[movie_idx]
print(list(movie_index[(P > 0.95) & (P < 1.0)]))