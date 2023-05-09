import torch
import CLImputeUtils as utils
import numpy as np

if __name__ == '__main__':

    device = torch.device('cpu:0')
    dataset_name = 'Zeisel'
    drop_rate=0.4

    ## Step1: reading dataset
    groundTruth_data, cells, genes = utils.load_data('data/Zeisel/Zeisel_top2000.csv')


    ## Step2: simulate dropout-events
    drop_data = utils.impute_dropout(groundTruth_data, drop_rate=drop_rate)
    print('dataset: {}, drop rate: {}'.format(dataset_name, drop_rate))

    ## Step3: training embedding
    X = torch.FloatTensor(np.copy(drop_data)).to(device)
    # Step3.1: loading a trained model
    model = utils.load_pretained_model(X, load_path='data/Zeisel/Zeisel_saved_model.pkl')
    # or Step3.2: training a model
    # model = utils.training(X, hidden_size=128, epoch=100, aug_rate=0.4)

    ## Step4: select k similiar cells and imputation
    choose_cell = utils.select_neighbours(model, X, k=20)
    # perform imputation
    imputed_data = utils.LS_imputation(drop_data, choose_cell, device)

    ## Step5: evaluation
    print('dropout data PCCs:', utils.pearson_corr(drop_data, groundTruth_data))
    print('imputed data PCCs:', utils.pearson_corr(imputed_data, groundTruth_data))
    print('dropout data L1:', utils.l1_distance(drop_data, groundTruth_data))
    print('imputed data L1:', utils.l1_distance(imputed_data, groundTruth_data))