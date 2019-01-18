from tools import *


if __name__ == '__main__':

    

    all_dict = {'maxvol': dict(),
         'pca': dict(),
         'random' : dict()
         }


    mnist_path = 'datasets/mnist'
    X_train, Y_train, X_test, Y_test = get_mnist(mnist_path)
    # X_train_maxvol, _, X_test_maxvol, _ = feature_selection(X_train, Y_train, X_test, Y_test, k = nfeat)
    
    # pca = PCA()

    # X_train_pca = pca.fit_transform(X_train)
    # X_test_pca  = pca.transform(X_test)
    # model_pca = Perceptron(max_iter=100, n_jobs=4)
    # model_pca.fit(X_train_pca, Y_train)
    # acc = accuracy_score(model_pca.predict(X_test_pca), Y_test)
    # print(acc)

    N_FEATURES = [20, 50, 100, 200, 300, 400, 500, 600, 700]
    # N_FEATURES = [200, 400, 600]
    for nfeat in N_FEATURES:

        # mnist_path = 'datasets/mnist'
        # X_train, Y_train, X_test, Y_test = get_mnist(mnist_path)
        X_train_maxvol, _, X_test_maxvol, _ = feature_selection(X_train, Y_train, X_test, Y_test, k = nfeat)

        model_maxvol = Perceptron(max_iter=100, n_jobs=4)
        print('Training on %s examples and %s featres' % X_train_maxvol.shape)
        model_maxvol.fit(X_train_maxvol, Y_train)

        maxvol_acc = accuracy_score(model_maxvol.predict(X_test_maxvol), Y_test)
        print('Maxvol feature selection acc:', maxvol_acc)
        all_dict['maxvol'][nfeat] = maxvol_acc

        pca = PCA(n_components=nfeat)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca  = pca.transform(X_test)

        model_pca = Perceptron(max_iter=100, n_jobs=4)
        print('Training on %s examples and %s featres' % X_train_pca.shape)
        model_pca.fit(X_train_pca, Y_train)
        pca_acc = accuracy_score(model_pca.predict(X_test_pca), Y_test)
        print('PCA feature selection acc:', pca_acc)
        all_dict['pca'][nfeat] = pca_acc

        random_scores = []
        for i in range(5):
            X_train_random, Y_train_random, X_test_random, Y_test_random = random_features(X_train, Y_train, X_test, Y_test, n_features=nfeat)
            print('Training on %s examples and %s featres' % X_train_random.shape)
            model_random = Perceptron(max_iter=100, n_jobs=4)
            model_random.fit(X_train_random, Y_train_random)
            random_acc = accuracy_score(model_random.predict(X_test_random), Y_test_random)
            random_scores.append(random_acc)
            # print('It %s:' % i, 'Random feature selection acc:' , random_acc)
        av_acc = sum(random_scores) / len(random_scores)
        print('Average random acc:', av_acc)
        all_dict['random'][nfeat] = av_acc

        print(all_dict)