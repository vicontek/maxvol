from tools import *

if __name__ == '__main__':
    K = 200
    # N_SAMPLES = 1000

    all_dict = {'maxvol': dict(),
                'random' : dict()
     }

    N_SAMPLES_list = [100, 200, 300, 400,  500, 1000, 1500, 2000, 3000]
    # K_list = [20, 50, 100, 300, 500, 700]

    mnist_path = 'datasets/mnist'
    X_train, Y_train, X_test, Y_test = get_mnist(mnist_path)

    # model_pca = Perceptron(max_iter=100, n_jobs=4)
    # model_pca.fit(X_train, Y_train)
    # acc = accuracy_score(model_pca.predict(X_test), Y_test)
    # print('PCA acc:', acc)


    rank_K = 200
    # for NS in N_SAMPLES_list:
    for NS in N_SAMPLES_list:

        
        X_train_maxvol, Y_train_maxvol = svd_mavol_sampling(X_train, Y_train, k=rank_K, n_samples=NS)
        print('Training on %s examples and %s featres' % X_train_maxvol.shape)
        model_sample = Perceptron(max_iter=100, n_jobs=4)
        model_sample.fit(X_train_maxvol, Y_train_maxvol)
        acc = accuracy_score(model_sample.predict(X_test), Y_test)
        print('Maxvol sampling acc:', acc)
        all_dict['maxvol'][NS] = acc


        random_scores = []
        for i in range(5):
            X_train_random, Y_train_random = random_sampling(X_train, Y_train, n_samples=NS * 10)
            print('Training on %s examples and %s featres' % X_train_random.shape)
            model_random = Perceptron(max_iter=100, n_jobs=4)
            model_random.fit(X_train_random, Y_train_random)
            acc = accuracy_score(model_random.predict(X_test), Y_test)
            random_scores.append(acc)
            print('It %s:' % i, 'random sampling acc:' , acc)
        print('Average random acc:', sum(random_scores) / len(random_scores))
        av_acc = sum(random_scores) / len(random_scores)
        all_dict['random'][NS] = av_acc

        print(all_dict)