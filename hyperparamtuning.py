#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import re
import pandas as pd
import datetime
import os
import warnings
from tqdm import tqdm
import sys
import ast




from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import make_scorer

from joblib import dump, load

date = datetime.datetime.today().strftime('%Y-%m-%d')


classifiershyperparam_cont= [
    (LinearSVR(random_state=0, max_iter=1500), {
    'epsilon': [0, 1],
    'loss': ['epsilon_insensitive','squared_epsilon_insensitive'],
    'C': [1.5, 1, 0.5, 0.2]}),
    (LinearRegression(), {
         'fit_intercept': [True]}),
    (ElasticNet(random_state=0), {
        'alpha': [3, 2, 1, 0.1],
        'l1_ratio': [0.8, 0.5, 0.2]}),
    (Ridge(random_state=0), {
    'solver': ['auto', 'svd','saga']}),
    (Lasso(random_state=0), {
        'alpha': [3, 2, 1, 0.1]}),
    (BayesianRidge(), {
    'tol': [0.01, 0.001, 0.0001],
    'alpha_1': [0.000001],
     'alpha_2': [0.000001],
     'lambda_1': [0.000001],
     'lambda_2':[0.000001]}),
    (KNeighborsRegressor(n_jobs= -1), {
    'n_neighbors': [25, 15, 10, 5],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree','kd_tree','brute','auto'],
    'leaf_size': [30, 20, 10, 5]}),
    (RandomForestRegressor(random_state=0, n_jobs= -1), {
    'n_estimators': [300],
    'min_samples_split': [100, 50, 20, 10],
    'min_samples_leaf': [50, 25, 10, 5],
    'max_features': ['auto', 'sqrt', 'log2']}),
    (AdaBoostRegressor(random_state=0), {
    'n_estimators': [300],
    'learning_rate': [1],
    'loss': ['linear', 'square', 'exponential'],
    'base_estimator': [RandomForestRegressor(random_state=0,
                                             n_jobs=-1,
                                             min_samples_split=5,
                                             min_samples_leaf=2,
                                             max_features='auto')]}),
    (GradientBoostingRegressor(random_state=0), {
    'n_estimators': [300],
    'learning_rate': [1],
    'loss': ['ls', 'lad', 'huber'],
    'subsample': [1, 0.8, 0.5],
    'min_samples_split': [100, 50, 20, 10],
    'min_samples_leaf': [50, 25, 10, 5],
    'max_depth': [20, 10, 5, 3]})
]

classifiers_cont = [
    LinearSVR(random_state=0, max_iter=1500),
    LinearRegression(),
    ElasticNet(random_state=0),
    Ridge(random_state=0),
    Lasso(random_state=0),
    BayesianRidge(),
    KNeighborsRegressor(n_jobs= -1),
    RandomForestRegressor(random_state=0, n_jobs= -1),
    AdaBoostRegressor(random_state=0,
    base_estimator = RandomForestRegressor(random_state=0,
                                             n_jobs=-1,
                                             min_samples_split=5,
                                             min_samples_leaf=2,
                                             max_features='auto')),
    GradientBoostingRegressor(random_state=0)
]

hyperparams_by_label = {
    'symptom-cont': classifiershyperparam_cont
}

classifiers_by_label = {
    'symptom-cont': classifiers_cont,
}


def hypertunepipe(dfd,
                  PATH,
                  date,
                  missing='nomiss',
                  outcome=False,
                  classifierhyperparams=False,
                  PCAUSE=False,
                  pcaoptions=False,
                  module_name=False):

    bestfit = pd.DataFrame(columns=['dataset', 'shape', 'outcome',
                                      'pca', 'pcaparams',
                                      'classifier', 'bestparams',
                                      'meanscore','stdscore'])
    for dataset, df in tqdm(dfd.iteritems(), total=len(dfd)):
        if classifierhyperparams:
            hyperparamsforlabel = classifierhyperparams
        elif outcome:
            hyperparamsforlabel = hyperparams_by_label[outcome]
        else:
            outcome = [part for part in dataset.split('_')
                               if part in hyperparams_by_label.keys()][0]
            hyperparamsforlabel = hyperparams_by_label[outcome]
        print '\n Processing {}'.format(dataset)

        # Splitting into features and outcome 
        X, y = (df.loc[:, df.columns != 'outcome'],
                df.loc[:, df.columns == 'outcome'].values.ravel())
        if PCAUSE:
            print 'Using PCA reduction', PCAUSE
            if (PCAUSE != 'linear' and PCAUSE != 'kernel'):
                sys.exit("PCA not properly defined")
            else:
                pass
        for classifier, classifierparam in hyperparamsforlabel:
            fit = classifier
            # get classifier name
            try:
                # get name of quantile regression o.w. just get
                # classifier name
                if classifierparam['loss'][0] == 'quantile':
                    clfname = '{}_{}'.format(
                        type(classifier).__name__,
                        ''.join(
                            str(classifierparam['alpha']).split('.')
                        ))
                else:
                    clfname = type(classifier).__name__
            except KeyError:
                clfname = type(classifier).__name__
            print '\n Searching for best params of', type(fit).__name__
            clf = GridSearchCV(fit, classifierparam, n_jobs=-1, cv=10, # 10 fold
                               error_score=0.0)

            if PCAUSE == 'linear':  # must align with the passed along hyperparams
                pca = PCA(**pcaoptions)
                Xpca = pca.fit_transform(X)
                # save pca model for later use on test set
                if not os.path.exists(
                        os.path.join(PATH, '..',
                                       'datasets', 'processed',
                                             'study1', 'PCA')):
                    os.makedirs(os.path.join(PATH, '..',
                                       'datasets', 'processed',
                                             'study1', 'PCA'))
                try:
                    dump(pca,
                         os.path.join(
                             PATH, '..',
                             'datasets', 'processed',
                             'study1', 'PCA',
                             '{}_{}_{}.joblib'.format(date,
                                                      dataset,
                                                      'pca_transform_linear')))
                except NameError as e:
                    print e
                    print 'continuing'
                    pass

                clf.fit(Xpca, y)
                print('Best parameters found:\n', clf.best_params_)
                bestfit = bestfit.append(
                    {'dataset': dataset,
                     'shape': Xpca.shape,
                     'outcome': outcome,
                     'pca': str(PCAUSE),
                     'pcaparams': '{}'.format(pcaoptions),
                     'classifier': type(fit).__name__,
                     'bestparams': clf.best_params_,
                     'meanscore': clf.best_score_,
                     'stdscore': clf.cv_results_['std_test_score'][clf.best_index_]},
                    ignore_index=True)

            if PCAUSE == 'kernel':
                print 'Processing number of components for kernel PCA'

                for run, ncomponents in enumerate([1, 2, 5, 10, 15, 25, 50, 100]):
                    print 'run', run
                    pcaoptions['n_components'] = ncomponents
                    pca = KernelPCA(**pcaoptions)
                    Xpca = pca.fit_transform(X)
                    clf.fit(Xpca, y)
                    means = clf.best_score_
                    stds = (
                        clf.cv_results_['std_test_score'][clf.best_index_])
                    bestParams = clf.best_params_
                    XpcaShape = Xpca.shape
                    bestpcaoptions = pcaoptions
                    if run == 0:
                        bestfit = bestfit.append(
                            {'dataset': dataset,
                             'shape': XpcaShape,
                             'outcome': outcome,
                             'pca': str(PCAUSE),
                             'pcaparams': '{}'.format(bestpcaoptions),
                             'classifier': type(fit).__name__,
                             'bestparams': bestParams,
                             'meanscore': means,
                             'stdscore': stds}, ignore_index=True)
                    else:
                        # If we have means since earlier
                        # Compare with previous means if present is better
                        # that means better score -> update our preferences
                        if means > bestfit.iloc[-1]['meanscore']:
                            bestfit.iloc[-1] = [dataset,
                                                XpcaShape,
                                                outcome,
                                                str(PCAUSE),
                                                '{}'.format(bestpcaoptions),
                                                type(fit).__name__,
                                                bestParams,
                                                means,
                                                stds]
                        else:
                            # keep previous if not
                            pass
                print('Best parameters found: \n kernel pca n = ',
                      bestfit.iloc[-1]['pcaparams'],
                      '\n with best params being \n ',
                      bestfit.iloc[-1]['bestparams'])

                # saving the pca object for later transformation with best
                # params
                try:
                    bestpcaoptionsdict = ast.literal_eval(
                        bestfit.iloc[-1]['pcaparams'])
                except ValueError as e:
                    # This happens for model which initiates other models
                    #  such as Adaboost which param dictionary initiates a RF model
                    bestpcaoptionsdict = eval(
                        bestfit.iloc[-1]['pcaparams'])
                pcak = KernelPCA(**bestpcaoptionsdict)
                pcak.fit_transform(X)
                if not os.path.exists(
                        os.path.join(PATH, '..',
                                       'datasets', 'processed',
                                             'study1', 'PCA')):
                    os.makedirs(os.path.join(PATH, '..',
                                       'datasets', 'processed',
                                             'study1', 'PCA'))
                try:
                    dump(pcak,
                         os.path.join(
                             PATH, '..',
                             'datasets', 'processed',
                             'study1', 'PCA',
                             '{}_{}_{}_{}.joblib'.format(date,
                                                      dataset, clfname,
                                                      'pca_transform_kernel')))
                except NameError as e:
                    print e
                    print 'continuing'
                    pass

            if PCAUSE == False:
                pcaoptions = str(None)
                clf.fit(X, y)
                print('Best parameters found:\n', clf.best_params_)
                bestfit = bestfit.append(
                    {'dataset': dataset,
                     'shape': X.shape,
                     'outcome': outcome,
                     'pca': str(PCAUSE),
                     'pcaparams': '{}'.format(pcaoptions),
                     'classifier': clfname,
                     'bestparams': clf.best_params_,
                     'meanscore': clf.best_score_,
                     'stdscore': clf.cv_results_['std_test_score'][clf.best_index_]}, ignore_index=True)
    if not os.path.exists(os.path.join(PATH, '..', 'results','study1')):
        print 'results folder does not exist, creating folder'
        os.makedirs(os.path.join(PATH, '..', 'results','study1'))
    with open(os.path.join(PATH, '..', 'results','study1',
                           '{}_{}_MLHypertune_results_{}.csv'.
            format(date,
                   module_name,
                   'pca-'+str(PCAUSE))), 'wb') as output:
        bestfit.to_csv(path_or_buf=output,
                       encoding='utf8',
                       index=False,
                       mode='wb')
    return bestfit