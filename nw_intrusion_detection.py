import pandas as pd
import numpy as np
from time import time
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler  # install scipy package
from sklearn.metrics import accuracy_score
from sklearn import cross_validation, neighbors
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import cross_val_score
import pickle

class IntrusionDetector:

    def __init__(self, data_path):
        self.kdd_path = data_path
        self.kdd_data = []
        self.kdd_numeric = []
        self.kdd_binary = []
        self.kdd_nominal = []
        self.kdd_label_2classes = []
        #read data from file
        self.get_data()


    def get_data(self):
        col_names = ["duration","protocol_type","service","flag","src_bytes",
            "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
            "logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
            "is_host_login","is_guest_login","count","srv_count","serror_rate",
            "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
        self.kdd_data = pd.read_csv(self.kdd_path, header=None, names = col_names)
        # print(kdd_data.describe())
        #print (kdd_data.shape)

    # To reduce labels into "Normal" and "Attack"
    def get_2classes_labels(self):
        label_2class = self.kdd_data['label'].copy()
        label_2class[label_2class != 'normal.'] = 'attack.'
        self.kdd_label_2classes = label_2class.values.reshape((label_2class.shape[0], 1))

    def preprocessor(self):
        # prepare 2 classes label for "attack" and "normal"
        self.get_2classes_labels()

        nominal_features = ["protocol_type", "service", "flag"]  # [1, 2, 3]
        binary_features = ["dst_bytes", "num_failed_logins", "num_compromised",\
                           "root_shell", "num_outbound_cmds", "is_host_login"]  # [6, 11, 13, 14, 20, 21]
        numeric_features = [
            "duration", "src_bytes",
            "land", "wrong_fragment", "urgent", "hot",
            "logged_in", "su_attempted", "num_root",
            "num_file_creations", "num_shells", "num_access_files",
            "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]

        #convert nominal features to numeric features
        #nominal features: ["protocol_type", "service", "flag"]
        self.kdd_nominal = self.kdd_data[nominal_features].stack().astype('category').unstack()

        self.kdd_nominal = np.column_stack((self.kdd_nominal["protocol_type"].cat.codes, \
                                               self.kdd_nominal["service"].cat.codes, \
                                               self.kdd_nominal["flag"].cat.codes))
        #print (kdd_nominal_encoded)

        self.kdd_binary = self.kdd_data[binary_features]

        # Standardizing and scaling numeric features
        self.kdd_numeric = self.kdd_data[numeric_features].astype(float)
        self.kdd_numeric = StandardScaler().fit_transform(self.kdd_numeric)


    def feature_reduction_PCA(self):
        #compute Eigenvectors and Eigenvalues
        mean_vec = np.mean(self.kdd_numeric, axis=0)
        cov_mat = np.cov((self.kdd_numeric.T))

        # Correlation matrix
        cor_mat = np.corrcoef((self.kdd_numeric.T))
        eig_vals, eig_vecs = np.linalg.eig(cor_mat)

        # To check that the length of eig_vectors is 1
        for ev in eig_vecs:
            np.testing.assert_array_almost_equal(1.0,np.linalg.norm(ev))
        #print ('eigen_vector length is 1')

        #to rank the eigenvalues from highest to lowest in order choose the top k eigenvectors and ignore the rest
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

        # Sort and print the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort()
        eig_pairs.reverse()
        #for i in eig_pairs:
        #    print(i[0])

        #feature reduction
        # just the 10 first items are greater 1 and one which is close to 1 => pick 11
        matrix_w = np.hstack((eig_pairs[0][1].reshape(32,1),
                              eig_pairs[1][1].reshape(32,1),
                              eig_pairs[2][1].reshape(32,1),
                              eig_pairs[3][1].reshape(32,1),
                              eig_pairs[4][1].reshape(32,1),
                              eig_pairs[5][1].reshape(32,1),
                              eig_pairs[6][1].reshape(32,1),
                              eig_pairs[7][1].reshape(32,1),
                              eig_pairs[8][1].reshape(32,1),
                              eig_pairs[9][1].reshape(32,1),
                              eig_pairs[10][1].reshape(32,1)))
        # projection to new feature space
        self.kdd_numeric = self.kdd_numeric.dot(matrix_w)
        '''
        from sklearn.decomposition import PCA as sklearnPCA
        sklearn_pca = sklearnPCA(n_components=10)
        self.kdd_numeric = sklearn_pca.fit_transform(self.kdd_numeric)
        '''

    def bayes_classifier(self):
        data = np.concatenate([self.kdd_numeric, self.kdd_binary, self.kdd_nominal, self.kdd_label_2classes], axis=1)

        # shuffle is by default = True
        kdd_train_data, kdd_test_data = train_test_split(data, train_size=0.8)

         #Load classifier from Pickle
        model=pickle.load(open("naivebayes.pickle", "rb"))

        # model = GaussianNB()
        # Train the model using the training sets
        #model.fit(kdd_train_data[:, :-1], kdd_train_data[:, -1])
        #with open('naivebayes.pickle','wb') as f:
        #     pickle.dump(model,f)

        # Predict
        predicts = model.predict(kdd_test_data[:, :-1])
        print ("Naive classifier:")
        accuracy = accuracy_score(kdd_test_data[:, -1], predicts)
        print ("Accuracy: ",accuracy)
        con_matrix = confusion_matrix(kdd_test_data[:, -1], predicts, labels=["normal.", "attack."])
        print("confusion matrix:")
        print(con_matrix)
        scores=cross_val_score(model,kdd_train_data[:, :-1], kdd_train_data[:, -1],cv=5)
        print('Cross validation Score:',scores)


    def knn_classifier(self):
        data = np.concatenate([self.kdd_numeric, self.kdd_label_2classes], axis=1)
        kdd_train_data, kdd_test_data = train_test_split(data, test_size=0.2)

        #Load classifier from Pickle
        clf=pickle.load(open("knearestneighbor.pickle", "rb"))
        # clf = neighbors.KNeighborsClassifier()
        # clf.fit(kdd_train_data[:, :-1], kdd_train_data[:, -1])
        # with open('knearestneighbor.pickle','wb') as f:
        #     pickle.dump(clf,f)
        print('model trained')
        predicts = clf.predict(kdd_test_data[:, :-1])
        accuracy=accuracy_score(kdd_test_data[:, -1], predicts)
        print('knn accuracy',accuracy)
        scores=cross_val_score(clf, kdd_train_data[:, :-1], kdd_train_data[:, -1],cv=5)
        print('Cross validation Score:',scores)

    def svm_classifier(self):
        data = np.concatenate([self.kdd_numeric, self.kdd_label_2classes], axis=1)
        kdd_train_data, kdd_test_data = train_test_split(data, train_size=0.1)

        # Create SVM classification object
        model = svm.SVC(kernel='rbf', C=0.2, gamma=1)
        model.fit(kdd_train_data[:, :-1], kdd_train_data[:, -1])

        # Predict Output
        predicts = model.predict(kdd_test_data[:, :-1])
        print("SVM classifier:")
        accuracy = accuracy_score(kdd_test_data[:, -1], predicts)
        print("Accuracy: ", accuracy)
        con_matrix = confusion_matrix(kdd_test_data[:, -1], predicts, labels=["normal.", "attack."])
        print("confusion matrix:")
        print(con_matrix)
        scores=cross_val_score(model,kdd_train_data[:, :-1], kdd_train_data[:, -1],cv=5)
        print('Cross validation Score:',scores)


    def decision_tree_classifier(self):
        #data = np.concatenate([self.kdd_numeric, self.kdd_label_2classes], axis=1)
        data = np.concatenate([self.kdd_numeric, self.kdd_binary, self.kdd_nominal, self.kdd_label_2classes], axis=1)


        kdd_train_data, kdd_test_data = train_test_split(data, train_size=0.8)

        model = tree.DecisionTreeClassifier()
        model.fit(kdd_train_data[:, :-1], kdd_train_data[:, -1])
        predicts = model.predict(kdd_test_data[:, :-1])
        print("Decision Tree classifier:")
        accuracy = accuracy_score(kdd_test_data[:, -1], predicts)
        print("Accuracy: ", accuracy)
        con_matrix = confusion_matrix(kdd_test_data[:, -1], predicts, labels=["normal.", "attack."])
        print("confusion matrix:")
        print(con_matrix)
        scores=cross_val_score(model,kdd_train_data[:, :-1], kdd_train_data[:, -1],cv=5)
        print('Cross validation Score:',scores)


def main():
    # Data path
    cwd = os.getcwd()  # current directory path
    kdd_data_path = cwd + "/data/kddcup.data_10_percent/kddcup.data_10_percent_corrected"

    i_detector = IntrusionDetector(kdd_data_path)
    i_detector.preprocessor()
    i_detector.feature_reduction_PCA()

    while (True):
        print("\n\n")
        print("1. Naive Bayes Classifier")
        print("2. SVM Classifier")
        print("3. KNN Classifier")
        print("4. Decision Tree Classifier")
        print("5. Quit")
        option = input("Please enter a value:")

        if (option == "1"):
             i_detector.bayes_classifier()
        elif option == "2":
             i_detector.svm_classifier()
        elif option == "3":
             i_detector.knn_classifier()
        elif option == "4":
             i_detector.decision_tree_classifier()
        elif option == "5":
             break


if __name__ == '__main__':
    main()
