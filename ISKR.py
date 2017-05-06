
"""
ISKR

Implemented by Sina Fathi-Kazerooni
May 2017

This project uses Scikit-Learn library and examples for text clustering: 
http://scikit-learn.org/stable/auto_examples/text/document_clustering.html
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

import sys
from time import time
import numpy as np
import collections

from terminaltables import AsciiTable

dataset = fetch_20newsgroups(subset='all', categories=None,shuffle=True, random_state=42)
#########################################################################
labels = dataset.target
countOfClusters = np.unique(labels).shape[0]
countOfClusters=3 # number of clusters
##########################################################################
#Search for user q 
myQuery=[]
if len(sys.argv)>1:
    myQuery.extend(sys.argv[1:])
else:
    a = input("Enter your query: ")
    if len(a)<2:
        a="jesus christ sin church jews day case shall"#"computer apple"# graphic microsoft"
        print("\nDefault query: %s"%a)
    a = a.lower()
    myQuery = a.split()
#print("\nExtracting features from the training dataset using a count vectorizer:\n")
t0 = time()
vectorizer = CountVectorizer(vocabulary=myQuery)
X = vectorizer.fit_transform(dataset.data)
print("n_samples: %d, n_features: %d" % X.shape)
########################################################################################
######### Create a new vectorized set with results all containing the user q  ##########
featureNames = vectorizer.get_feature_names()
newSet=[]
filenames=[]
counter=0
for doc in range(0,len(dataset.data)):
    if all(X[doc,i]>0 for i in range(0,len(myQuery))):
        counter+=1
        print(">>> Query {}".format(myQuery)," was found in Doc %d:\n"%doc)
        print("\n".join(dataset.data[doc].split("\n")[:3])) #Print first 3 lines of document
        print()
        newSet.append(dataset.data[doc])
        filenames.append(dataset.filenames[doc])
if counter<6:
    print("Not enough results were found for query: {}!\n\n".format(myQuery))
    exit()
########################################################################################
##### Run tf-idf vectorizer on new set (results from user q)
print("Extracting features from the training dataset using TF-IDF vectorizer.\n")
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                            min_df=0.2, stop_words='english',
                            use_idf=False)
X = vectorizer.fit_transform(newSet)
###############################################################################
# K-Means
km = KMeans(n_clusters=countOfClusters, init='k-means++', max_iter=100, n_init=1)
print("Clustering data with k-means algorithm.")
km.fit(X)

##### Uncomment below to show top terms in each cluster #####
#print("\nTop terms per cluster:")
#order_centroids = km.cluster_centers_.argsort()[:, ::-1]
#terms = vectorizer.get_feature_names()
#for i in range(countOfClusters):
#    print("Cluster %d:" % i, end='')
#    for ind in order_centroids[i, :10]:
#        print(' %s' % terms[ind], end='')
#    print()
#print("\n")

##### Uncomment below to show results in each cluster #####
#print("Results in each cluster:\n")
#for i in range(0,len(km.labels_)):
#    print(km.labels_[i], end=': ')
#    print("\n".join(newSet[i].split("\n")[:3]),end="\n\n")
#print("\n")

################################################################################
# Run ISKR
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
count = collections.Counter(km.labels_)
expandedQueryList=[]
t0 = time()
for i in range(0,countOfClusters):
    converged=False
    iterCount=0
    myNewQuery=[]
    myNewQuery.clear()
    myNewQuery.extend(myQuery)
    clusterTotal=count.get(i)
    otherClusterTotal=len(newSet)-count.get(i)    
    preQuery=[]
    while(converged==False and iterCount<100):
        preQuery.clear()
        preQuery.extend(myNewQuery)
        iterCount+=1
        print("\nIteration: {}".format(iterCount))
        print("Query: {}\n".format(myNewQuery))
        newSetCluster=[]
        newSetC=[]
        newSetOtherCluster=[]
        newSetOC=[]
        for n in range(0,len(km.labels_)):
            if km.labels_[n]==i:
                newSetCluster.append(newSet[n])
            else:
                newSetOtherCluster.append(newSet[n])
        #####Using the expanded query create a table cost/benefit/value
        table_data=[]
        table_data.clear()
        table_data = [['Keyword', 'Benefit', 'Cost', 'Value']]
        for ind in order_centroids[i, :3]:
            newSetC.clear()
            newSetOC.clear()
            myExpandedQuery=[]
            myExpandedQuery.extend(myNewQuery)
            expansionQ=terms[ind]
            if (terms[ind] not in myNewQuery):
                myExpandedQuery.append(expansionQ)
            tableRow=[]
            tableRow.append(expansionQ)
            vectorizer = CountVectorizer(vocabulary=myExpandedQuery)
            X = vectorizer.fit_transform(newSetCluster)
            for doc in range(0,len(newSetCluster)):
                if all(X[doc,x]>0 for x in range(0,len(myExpandedQuery))):                
                    newSetC.append(newSetCluster[doc])
            cost = clusterTotal-len(newSetC)
            
            X = vectorizer.fit_transform(newSetOtherCluster)
            for doc in range(0,len(newSetOtherCluster)):
                if all(X[doc,x]>0 for x in range(0,len(myExpandedQuery))):
                    newSetOC.append(newSetOtherCluster[doc])
            benefit = otherClusterTotal-len(newSetOC)
        
            tableRow.append("{}".format(benefit))
            tableRow.append("{}".format(cost))

            if cost!=0:
                value = benefit/cost    
                tableRow.append("%.2f"%value)
            elif benefit==0:
                value=0
                tableRow.append("%.2f"%value)
            else:
                tableRow.append("inf")
            table_data.append(tableRow)
            newSetC.clear()
            newSetOC.clear()
        table = AsciiTable(table_data)
        print(table.table)
        maxIndex=1
        maxVal = float(0)
        for num in range(1,len(table_data),1): #ignore the first row (title row)
            valueData=table_data[num][3]
            valueName=table_data[num][0]
            if(valueName not in myNewQuery):
                if valueData is "inf":
                    maxIndex=num
                    maxVal = 0
                elif float(valueData) > maxVal:
                    maxVal=float(valueData)
                    maxIndex=num   
        if maxVal<1 and maxVal!=0:
            converged=True
            expandedQueryList.append(myNewQuery)
        else:
            valueName=table_data[maxIndex][0]
            if(valueName not in myNewQuery): 
                myNewQuery.append(table_data[maxIndex][0])
            if(preQuery==myNewQuery):
                converged=True
                if (myNewQuery not in expandedQueryList):
                    expandedQueryList.append(myNewQuery)
print("\n")
print("Expanded queries:")
for line in range(0,len(expandedQueryList)):
    print(expandedQueryList[line][:])
print("\n\nDuration: %0.4fs\n\n\n\n"% (time() - t0))