#from kaggle user cclark

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('data/train.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('data/test.csv','r'), delimiter=',', dtype='f8')[1:]

    #build crossvalidation training set
    train_train, train_test, target_train, target_test = cross_validation.train_test_split(train, target, test_size=0.4, random_state=0)
    print train_train.shape
    print train_test.shape
    
    #create and train the random forest
    rf = RandomForestClassifier(n_estimators=50, n_jobs=2)
    rf.fit(train_train, target_train)
    print "crossval score is: ", rf.score(train_test, target_test)

    savetxt('data/submission.csv', rf.predict(test), delimiter=',', fmt='%f')

if __name__=="__main__":
    main()