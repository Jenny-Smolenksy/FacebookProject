
from neuralNetworkFB import NeuralNetworkFB
import numpy as np

def main():

        #preProcessing
        #load train set
        train_file = "data/train.csv"
        test_file = "data/test.csv"
        neural_net  = NeuralNetworkFB(train_file)
        #neural_net.cross_validation(20, 0.02, True, False, 0.005)
        #best_params = neural_net.check_hyper_parameters()
        # train with best hyper parameters - may be taken from best params
        neural_net.train_model(10, 0.02, True, False, 0.005)
        #neural_net.predict_test(test_file)

        data_x_test = np.loadtxt(test_file, skiprows=1, delimiter=';', usecols=range(0, 12))
        data_y_test = np.loadtxt(test_file, skiprows=1, delimiter=';', usecols=12)

        for i in range(len(data_x_test)):

            sample = data_x_test[i]
            real_tag = int(data_y_test[i])
            y_tag = neural_net.predict_sample_tag(sample)
            print(f"real tag : {real_tag}, prediction : {y_tag}")



if __name__ == '__main__':
    main()