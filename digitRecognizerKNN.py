#from kaggle user cclark

from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn import cross_validation
from sklearn.decomposition import RandomizedPCA
from numpy import genfromtxt, savetxt, hstack
import numpy as np

def main():
    #create the training & test sets, skipping the header row with [1:]
    print "Loading data..."
    dataset = genfromtxt(open('data/train.csv','r'), delimiter=',', dtype='u1')[1:]    
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('data/test.csv','r'), delimiter=',', dtype='u1')[1:]

    #build crossvalidation training set
    train_train, train_test, target_train, target_test = cross_validation.train_test_split(train, target, test_size=0.2, random_state=0)
    print train_train.shape
    print train_test.shape

    #PCA
    #pca = RandomizedPCA(n_components=100)
    #pca.fit(train_train)
    
    #create and train the random forest
    #rf = RandomForestClassifier(n_estimators=1000, n_jobs=4)
    #rf.fit(pca.transform(train_train), target_train)
    #print "crossval score is: ", rf.score(pca.transform(train_test), target_test)

    print "Training KNN Model..."
    #Crossval scores for n_neighbors: 4->0.9676, 5-> 0.9681, 6-> 0.9674, 10->0.965
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_train, target_train)
    print "crossval score is: ", knn.score(train_test, target_test)

    #Generate columns for the output
    print "Fitting the test set..."
    labelid = np.array(range(1,28001))
    output = knn.predict(test)

    #Save the data for submission
    print "Saving the submission file..."
    savetxt('data/submissionKNN.csv', np.column_stack((labelid, output)), delimiter=',', header="ImageId,Label", fmt='%u', comments='')

if __name__=="__main__":
    main()

