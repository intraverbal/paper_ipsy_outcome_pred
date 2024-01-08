#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import os
import datetime
import random
import warnings

# imports of prediction/pca prediction show identical functions with different naming for adaptions based on the format of the data handling
from prediction import * 
from pca_prediction import * 

from modules.datacleaner import savedfs, reloaddfs
from modules.readfile import readinstrumentsfile, readarbetsbladfile

from sklearn.model_selection import train_test_split

from modules.hyperparamtuning import hypertunepipe, hyperparams_by_label, classifiers_by_label

PATH = ''
CONTOUTCOME = True
date = datetime.datetime.today()
VERBOSE = True
MISSMIN = True
Modesofdata = ['Handpicked','PCA-linear','PCA-kernel'] # benchmark is built into this 
handpickfix = False
hyperparamyeshp = False 
analyseshp = True 
linearpcafix = True
hyperparamyeslpca = True
hyperparamyeskpca = True
analyseslpcal = True
analyseslpcak = True


if __name__ == "__main__":
    #hyperparamyes = raw_input('Do you want to hyperparam y/n? \n')
    #analyses = raw_input('Do you want to run analyses y/n? \n')
    try:
        hyperparamyes
    except NameError:
        hyperparamyes = 'y'
    try:
        analyses
    except NameError:
        analyses = 'y'
    # Handpicked/benchmark first

    # Import needed files
    if handpickfix:
        instruments = readinstrumentsfile(
            os.path.join(PATH, '..',
                         'datasets','instruments_filtered_merged_scored_labeled.csv'))
        arbetsblad = readarbetsbladfile(
            os.path.join(PATH, '..',
                         'datasets','arbetsblad_wide_interpreted.csv'), wide=True)
        outcome = readinstrumentsfile(os.path.join(PATH, '..',
                         'datasets','outcome.csv'))

        # Note that outcomedict have no effect if CONTOUTCOME = TRUE (which it is)
        dfdict, meanstdforoutcome, dflabeloutcome = wrapper(outcome,
                                                           arbetsblad,
                                                           instruments,
                                                           whatoutcome=outcomedict)
        # d names are now 'Treatment_timepoint-nahandling_benchmarkornobenchmark'
        # ex. Depression_screen_imputed, Depression_screen_imputed_benchmark

    modeofdata = 'Handpicked'

    if hyperparamyeshp:
        traindfdict = reloaddfs(regexfile=r'.*_.*week04.*.csv',
                                fullPATH=os.path.join(PATH, '..',
                                                      'datasets', 'processed',
                                                      'study1',
                                                      'Handpicked', 'train', 'missfix'))
        allbestparams = (
            hypertunepipe(traindfdict,
                          date=date.strftime('%Y-%m-%d'),
                          missing=False,
                          outcome='symptom-cont',
                          PATH=PATH,
                          classifierhyperparams=False,
                          PCAUSE=False,
                          pcaoptions=False,
                          module_name=modeofdata)
        )
    if analyseshp:
        traindfdict = reloaddfs(regexfile=r'.*_.*_.*.csv',
                                fullPATH=os.path.join(PATH, '..',
                                                      'datasets', 'processed',
                                                      'study1',
                                                      'Handpicked', 'train', 'missfix'))
        testdfdict = reloaddfs(regexfile=r'.*_.*_.*.csv',
                               fullPATH=os.path.join(PATH, '..',
                                                     'datasets', 'processed',
                                                     'study1',
                                                     'Handpicked', 'test', 'missfix'))
        dflabeloutcome = reloaddf(
            fullPATH=os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'Handpicked'),
        regexfile = r'.*dflabeloutcome.csv$')
        meanstdforoutcome = reloaddfs(
            fullPATH=os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'Handpicked'),
            regexfile=r'.*meanstdforoutcome.txt$',
            txttuple=True)
        # run analyses #also spits out the results
        try:
            paramfile = newest(os.path.join(PATH, '..', 'results','study1'),
                               r'.*{}_MLHypertune_results_.*.csv$'.format(
                                   'Handpicked'))
            print '--------------\n loading param file \n', paramfile
            allbestparams = pd.read_csv(paramfile, dtype=np.str_,
                                        encoding='utf8')
        except Exception as e:
            print 'No hyperparams'
            allbestparams = False

        runanalyses(dfd=traindfdict,
                    dfdtest=testdfdict,
                    PATH=PATH,
                    date=date.strftime('%Y-%m-%d'),
                    dflabeloutcome=dflabeloutcome,
                    meanstdforoutcome=meanstdforoutcome,
                    hyperparams=allbestparams,
                    classifiers_by_label=False,
                    in_production=False,
                    diagnostics=True,
                    modelbuilding=False,
                    PCAUSE=False,
                    pcaoptions=False,
                    anymiss=False,
                    savedf=False,
                    outcome='symptom-cont',
                    module_name=modeofdata)

    if linearpcafix:
        modeofdata = 'linearPCA'

        instruments = readinstrumentsfile(
            os.path.join(PATH, '..',
                         'datasets', 'instruments_filtered_merged_scored_labeled.csv'))
        arbetsblad = readarbetsbladfile(
            os.path.join(PATH, '..',
                         'datasets', 'arbetsblad_wide_interpreted.csv'), wide=True)
        outcome = readinstrumentsfile(os.path.join(PATH, '..',
                                                   'datasets', 'outcome.csv'))
        # No wrapper function here just the seperate functions run in the same order as in wrapper 
        # same function as in the wrapper function for handpicked modified to handle the pca 
        df, dflabeloutcome = mergedfssetupfeatures_pca(outcome,
                                                       arbetsblad,
                                                       instruments,
                                                       whatoutcome=outcomedict)
        df, meanstdforoutcome = merged_cleansymptoms_pca(df)
        dfdict_week = splitintoweekwise_pca(df)
        dfdict_week_treatment = splitintotreatments_pca(dfdict_week)
        dfdict_missfix = missingwrapper_pca(dfdict_week_treatment)
        
    if hyperparamyeslpca:
        traindfdict = reloaddfs(
            regexfile=r'.*_.*week04.*.csv',
            fullPATH=os.path.join(os.path.join(PATH, '..',
                                               'datasets', 'processed',
                                               'study1', 'PCA', 'train', 'missfix')))
        dflabeloutcome = reloaddf(
            fullPATH=os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'PCA'),
            regexfile=r'.*dflabeloutcome.csv$')
        meanstdforoutcome = reloaddfs(
            fullPATH=os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'PCA'),
            regexfile=r'.*meanstdforoutcome.txt$',
            txttuple=True)
        allbestparams = hypertunepipe(traindfdict,
                                      date = date.strftime('%Y-%m-%d'),
                                      missing='nomiss',
                                      outcome='symptom-cont',
                                      PATH=PATH,
                                      PCAUSE='linear',
                                      pcaoptions={
                                          'n_components': 0.95,
                                          'random_state': 0},
                                      module_name=modeofdata)
    if hyperparamyeskpca:
        traindfdict = reloaddfs(
            regexfile=r'.*_.*week04.*.csv',
            fullPATH=os.path.join(os.path.join(PATH, '..',
                                               'datasets', 'processed',
                                               'study1', 'PCA', 'train', 'missfix')))
        dflabeloutcome = reloaddf(
            fullPATH=os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'PCA'),
            regexfile=r'.*dflabeloutcome.csv$')
        meanstdforoutcome = reloaddfs(
            fullPATH=os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'PCA'),
            regexfile=r'.*meanstdforoutcome.txt$',
            txttuple=True)
        modeofdata = "kernelPCA"
        allbestparams = hypertunepipe(traindfdict,
                                      date=date.strftime('%Y-%m-%d'),
                                      missing='nomiss',
                                      outcome='symptom-cont',
                                      PATH=PATH,
                                      PCAUSE='kernel',
                                      pcaoptions={
                                          'n_components': None,
                                          'random_state': 0,
                                          'kernel': "rbf"},
                                      module_name=modeofdata)
    if analyseslpcal:
        # run analyses #also spits out the results
        # searchfor = 'True|linear|kernel'
        dflabeloutcome = reloaddf(
            fullPATH=os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'PCA'),
            regexfile=r'.*dflabeloutcome.csv$')
        meanstdforoutcome = reloaddfs(
            fullPATH=os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'PCA'),
            regexfile=r'.*meanstdforoutcome.txt$',
            txttuple=True)
        traindfdictl = reloaddfs(regexfile=r'.*_.*_.*.csv',
                                 fullPATH=os.path.join(PATH, '..',
                                                       'datasets',
                                                       'processed',
                                                       'study1',
                                                       'PCA',
                                                       'train',
                                                       'missfix'))
        testdfdictl = reloaddfs(regexfile=r'.*_.*_.*.csv',
                                 fullPATH=os.path.join(PATH, '..',
                                                       'datasets',
                                                       'processed',
                                                       'study1',
                                                       'PCA',
                                                       'test',
                                                       'missfix'))

        try:
            paramfile = newest(
                os.path.join(PATH, '..', 'results', 'study1'),
                r'.*{}_MLHypertune_results_.*.csv$'.format(
                    'linearPCA'))
            print '--------------\n loading param file \n', paramfile
            allbestparams = pd.read_csv(paramfile, dtype=np.str_,
                                        encoding='utf8')
        except Exception as e:
            print 'No hyperparams'
            allbestparams = False
        runanalyses(dfd=traindfdictl,
                    dfdtest=testdfdictl,
                    PATH=PATH,
                    date=date.strftime('%Y-%m-%d'),
                    dflabeloutcome=dflabeloutcome,
                    meanstdforoutcome=meanstdforoutcome,
                    hyperparams=allbestparams,
                    classifiers_by_label=False,
                    in_production=False,
                    diagnostics=True,
                    modelbuilding=False,
                    PCAUSE='linear',
                    pcaoptions={
                        'n_components': 0.95,
                        'random_state': 0},
                    anymiss=False,
                    savedf=False,
                    outcome='symptom-cont',
                    module_name='linearPCA')
    if analyseslpcak:
        # run analyses #also spits out the results
        # searchfor = 'True|linear|kernel'
        dflabeloutcome = reloaddf(
            fullPATH=os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'PCA'),
            regexfile=r'.*dflabeloutcome.csv$')
        meanstdforoutcome = reloaddfs(
            fullPATH=os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'PCA'),
            regexfile=r'.*meanstdforoutcome.txt$',
            txttuple=True)
        traindfdictk = reloaddfs(regexfile=r'.*_.*_.*.csv',
                                 fullPATH=os.path.join(PATH, '..',
                                                       'datasets',
                                                       'processed',
                                                       'study1',
                                                       'PCA',
                                                       'train',
                                                       'missfix'))
        testdfdictk = reloaddfs(regexfile=r'.*_.*_.*.csv',
                                fullPATH=os.path.join(PATH, '..',
                                                      'datasets',
                                                      'processed',
                                                      'study1',
                                                      'PCA',
                                                      'test',
                                                      'missfix'))

        try:
            paramfile = newest(
                os.path.join(PATH, '..', 'results', 'study1'),
                r'.*{}_MLHypertune_results_.*.csv$'.format(
                    'kernelPCA'))
            print '--------------\n loading param file \n', paramfile
            allbestparams = pd.read_csv(paramfile, dtype=np.str_,
                                        encoding='utf8')
        except Exception as e:
            print 'No hyperparams'
            allbestparams = False
        runanalyses(dfd=traindfdictk,
                    dfdtest=testdfdictk,
                    PATH=PATH,
                    date=date.strftime('%Y-%m-%d'),
                    dflabeloutcome=dflabeloutcome,
                    meanstdforoutcome=meanstdforoutcome,
                    hyperparams=allbestparams,
                    classifiers_by_label=False,
                    in_production=False,
                    diagnostics=True,
                    modelbuilding=False,
                    PCAUSE='kernel',
                    pcaoptions=False,
                    anymiss=False,
                    savedf=False,
                    outcome='symptom-cont',
                    module_name='kernelPCA')
# Done

