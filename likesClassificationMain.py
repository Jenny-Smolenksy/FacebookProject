from decisionTreeFB import DecisionTreeFB
from neuralNetworkFB import NeuralNetworkFB
import numpy as np

def main():

        #preProcessing
        #load train set
        train_file = "data/train.csv"
        test_file = "data/test.csv"


        neural_net  = NeuralNetworkFB(train_file)
        # train with best hyper parameters - may be taken from best params
        neural_net.train_model(10, 0.02, True, False, 0.005)
        #neural_net.cross_validation(20, 0.02, True, False, 0.005)
        #best_params_net = neural_net.check_hyper_parameters()
        #neural_net.predict_test(test_file) #all predictions

        decision_tree = DecisionTreeFB(train_file)
        #train with best
        decision_tree.train_tree("entropy", "random", 11, 0.01)
        #decision_tree.cross_validation()
        #best_params_tree = decision_tree.check_hyper_parameters()
        decision_tree.predict_test(test_file)

        data_x_test = np.loadtxt(test_file, skiprows=1, delimiter=';', usecols=range(0, 12))
        data_y_test = np.loadtxt(test_file, skiprows=1, delimiter=';', usecols=12)

        for i in range(len(data_x_test)):

            sample = data_x_test[i]
            real_tag = int(data_y_test[i])
            y_tag_net = neural_net.predict_sample_tag(sample)
            y_tag_tree = decision_tree.predict_sample(sample)
            print(f"real tag : {real_tag}, prediction net : {y_tag_net}, prediction tree : {y_tag_tree}")



if __name__ == '__main__':
    main()