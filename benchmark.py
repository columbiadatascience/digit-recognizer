#from kaggle user cclark

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.decomposition import RandomizedPCA
from numpy import genfromtxt, savetxt, hstack
import numpy as np

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('data/train.csv','r'), delimiter=',', dtype='u1')[1:]    
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('data/test.csv','r'), delimiter=',', dtype='u1')[1:]

    #build crossvalidation training set
    train_train, train_test, target_train, target_test = cross_validation.train_test_split(train, target, test_size=0.2, random_state=0)
    print train_train.shape
    print train_test.shape

    #PCA
    pca = RandomizedPCA(n_components=40)
    pca.fit(train_train)
    
    #create and train the random forest
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=4)
    rf.fit(hstack((train_train, pca.transform(train_train))), target_train)
    print "crossval score is: ", rf.score(hstack((train_test, pca.transform(train_test))), target_test)

    labelid = np.array(range(1,28001))

    output = rf.predict(hstack((test, pca.transform(test))))
    savetxt('data/submission.csv', np.column_stack((labelid, output)), delimiter=',', header="ImageId,Label", fmt='%u', comments='')

if __name__=="__main__":
    main()

