#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import re
import pandas as pd
import datetime
import os
import warnings
import sys
import ast

from modules.readfile import readarbetsbladfile
from modules.readfile import readinstrumentsfile
from modules.datacleaner import mainmerger
from modules.datacleaner import savedfs, reloaddfs, newest
from modules.datacleaner import writefile
from modules.datacleaner import convertdatescols
from modules.datacleaner import labeldictionary, labels_cont, labels_bin, weekly_split_dict


from joblib import dump, load
from tqdm import tqdm
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from missingpy import MissForest

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, confusion_matrix


from modules.hyperparamtuning import *



PATH = ''
CONTOUTCOME = True
date = datetime.datetime.today()

##############################################################################
# Merging functions
##############################################################################

def mergedfssetupfeatures(outcome,arbetsblad,instruments,whatoutcome=outcomedict):
    """
    1. Minimzing the features and creating main symptoms.
    Furthermore creating the datasets
    :param outcome: outcomedf
    :param arbetsblad: arbetsblad_wide_interpreted
    :param instruments: instruments_filtered_merged_scored_labeled
    :return: a df with above dataframes fused into one
    """
    print 'Merging outcome,arbetsblad and instruments'
    todrop = []
    # fix some col. names #these neeed to exist in the namespace
    df = cleanoutcome(outcome) #get out everything we need and dropsome

    # Rename treatments first
    df['Treatment'] = df['Treatment'].replace(
        {'Depression STARTA EJ': 'Depression',
         'Depression 2.0': 'Depression',
         'Paniksyndrom': 'Panic',
         'Social fobi STARTA EJ': 'Social_Anxiety',
         'Social fobi 2.0': 'Social_Anxiety'})

    # fixing upp sex
    df['sex'] = df['sex'].map({'M':1,'F':0})

    # Make labels and treatment indicatiors

    # Treatment indicator
    #
    # if u wanna keep 'Treatment' org col 4 later
    df.loc[:,'Treat_org_drop'] = df.loc[:,'Treatment']
    df = pd.get_dummies(df,
                   columns=['Treatment'],
                   drop_first=False, prefix='Treatment')
    df.columns = [col.replace('Treatment_','') for col in df.columns]

    # labelfix
    df['outcome'] = np.nan
    # Label
    if CONTOUTCOME:
        df.loc[df['Treat_org_drop'] == 'Panic', 'outcome'] = (
            df.loc[df['Treat_org_drop'] == 'Panic', 'PDSS-SR-3064_POST_sum'])
        df.loc[df['Treat_org_drop'] == 'Depression', 'outcome'] = (
            df.loc[df['Treat_org_drop'] == 'Depression', 'MADRS-1951_POST_sum'])
        df.loc[df['Treat_org_drop'] == 'Social_Anxiety', 'outcome'] = (
            df.loc[df['Treat_org_drop'] == 'Social_Anxiety', 'LSAS-2241_POST_sum'])
        # save dictionary outcome for later scoring properties - deprecated 
        df.loc[df['Treat_org_drop'] == 'Panic', 'outcomesave'] = (
            df.loc[df['Treat_org_drop'] == 'Panic', 'PDSS-SR-3064_label7'])
        df.loc[df['Treat_org_drop'] == 'Depression', 'outcomesave'] = (
            df.loc[df['Treat_org_drop'] == 'Depression', 'MADRS-1951_label10'])
        df.loc[df['Treat_org_drop'] == 'Social_Anxiety', 'outcomesave'] = (
            df.loc[df['Treat_org_drop'] == 'Social_Anxiety', 'LSAS-2241_label34'])
    else: # deprecated 
        df.loc[df['Treat_org_drop'] == 'Panic', 'outcome'] = (
            df.loc[df['Treat_org_drop'] == 'Panic', 'PDSS-SR-3064_label7'])
        df.loc[df['Treat_org_drop'] == 'Depression', 'outcome'] = (
            df.loc[df['Treat_org_drop'] == 'Depression', 'MADRS-1951_label10'])
        df.loc[df['Treat_org_drop'] == 'Social_Anxiety', 'outcome'] = (
            df.loc[df['Treat_org_drop'] == 'Social_Anxiety', 'LSAS-2241_label34'])

    # Creating number of comorbidities
    # don't include the specifiers
    comdiag = r'(.*0_PRE_0_comdiag-)(?!other[A-Z]-spec$)'
    comorbidcols = [x for x in list(outcome) if re.search(comdiag, x)]
    masktoignore = (df.loc[:, comorbidcols].astype(float).isna().sum(axis=1) > 16)
    df.loc[:,'ncomorbid'] = (  # summing all comorbid
        df.loc[~masktoignore, comorbidcols].astype(float).sum(axis=1))


    # Extending what we drop because we summarised these columns
    todrop.extend([x for x in df.columns if 'label' in x])
    todrop.extend(['Treatment'])
    todrop.extend(comorbidcols)

    # create outcome 
    if CONTOUTCOME:
        
        print 'Using', 'continous symptom', 'as dictoutcome'
        print ('Not removing those with unknown outcome n=',
               df['outcome'].isnull().sum())
        #df = df.loc[~df['outcome'].isnull(), :]
        df['outcomesave'] = df['outcomesave'].replace(outcomedict)

        # Generating a seperate dataframe to keep track of both continous
        # and the dichotomized outcome for later labling
        colforlabling = ['Patient', 'Treat_org_drop','outcome','outcomesave']
        moreforlabel = [x for x in df.columns if re.search(
            r'(1951|2241|3064)_(SCREEN|PRE)_sum',x)]
        colforlabling.extend(moreforlabel)
        dflabeloutcome = df.loc[:, colforlabling]
        dflabeldrop = []
        # Must generate a main symptom for each treatment (not standardized)
        # because it is not used in the model
        for i in ['PRE']:
            for j in ['sum']:
                dflabeloutcome.loc[:, 'mainsymptom_{}_{}'.format(i, j)] = np.nan
                dflabeloutcome.loc[:, 'mainsymptom_{}_{}'.format(i, j)] = (
                    dflabeloutcome.apply(mainmerger, axis=1, args=[i, j]))
                dflabeldrop.extend(['PDSS-SR-3064_{}_{}'.format(i, j),
                               'MADRS-1951_{}_{}'.format(i, j),
                               'LSAS-2241_{}_{}'.format(i, j)])
        # need patient to be index to find right patient later in the
        # analyses
        dflabeloutcome = dflabeloutcome.set_index('Patient',drop=True)
        dflabeloutcome.drop(dflabeldrop,inplace=True,axis='columns')
        df.drop('outcomesave', inplace=True, axis='columns')
    else:
        print 'Using', whatoutcome, 'as dictoutcome'
        df['outcome'] = df['outcome'].replace(whatoutcome)
        print ('NOT removing those with unknown outcome n=',
            len(df[df['outcome']=='unknown']) )
        #df = df.loc[df['outcome'] != 'unknown',:]

    # do not remove

    # Gathering info from other sources (arbetsblad and instruments)

    # Arbetsblad

    pickedarbetsblad = cleanarbetsbladen(arbetsblad)
    df = df.merge(pickedarbetsblad, how='left', on=u'Patient')

    # instruments
    pickedinstruments, namesfrominstruments = cleaninstruments(instruments)
    df = df.merge(pickedinstruments, how='inner', on = u'Patient')

    #rename for ease of understanding + feature engineer for understanding
    df = df.rename(
        columns={'Anamnes-(ur-SCID)-1827_SCREEN_1846_7': 'currentwork_proff'})

    namesfrominstruments = namesfrominstruments.remove(u'Patient')
    # get dummy coding for the categories of marital and education
    df = pd.get_dummies(df,
                        columns=[u'Anamnes-(ur-SCID)-1827_SCREEN_1833_2a',
                                    u'Anamnes-(ur-SCID)-1827_SCREEN_1843_5'],
                        drop_first=True,prefix=['Marital_1833','Edu_1843'])
    # Set index before all the splitting begins
    df.set_index('Patient', inplace=True)

    # Fixing cscale # fill upp missing with the other variables
    df['cscale'] = df['C-skalan-depression-112598_PRE_sum'].fillna(
        df['C-skalan-depression-112598_MID_sum']).fillna(
        df['C-skalan-panik-61991_MID_sum']).fillna(
        df['C-skalan-social-fobi-klinik-518684_MID_sum'])
    # then remove
    todrop.extend([x for x in df.columns if re.search(r'^C-skalan',x)])

    print 'Done!'
    if dflabeloutcome is None:
        dflabeloutcome = pd.DataFrame()
    if not os.path.exists(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1',
                                       'Handpicked')):
        print 'Result folder does not exist, creating results folder'
        os.makedirs(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1',
                                       'Handpicked'))
    with open(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1',
                                       'Handpicked',
              '{}_{}.csv'.format(date.strftime('%Y-%m-%d'),
                                           'dflabeloutcome')), 'wb') as output:
        dflabeloutcome.to_csv(path_or_buf=output,
                  encoding='utf8',
                  index=True,
                  mode='wb')
    return df.drop(todrop,axis='columns'), dflabeloutcome

def merged_cleansymptoms(df):
    """
    takes in a merged dataframe (from outcome,instruments,arbetsblad) and
    cleans the symptoms and merge them into one - corresponding to the
    right main symptom of that treatment
    ALSO SCALES the instruments so they can be fused (right now with standard
    scaler - ergo ~N(0,1)
    :param df:
    :return:
    """
    # Do the symtom cleaning while everyone is in one big dataset because u
    # can start norming in the individual columns then fuse them together pretty
    # simply - include duration in this .
    todrop = []

    # Begin by standardising columns
    listtostand = [x for x
                   in df.columns
                   if re.search(r'(1951|2241|3064|age|messages|homeworks|cscale)',
                                x)]
    listtostand = [x for x in listtostand if not 'DateCompleted' in x
                   and not 'POST' in x]

    df[listtostand] = StandardScaler().fit_transform(df[listtostand])

    if CONTOUTCOME:
        # Standardising the outcome to make all treatments comparable
        df.loc[:, 'outcome'] = pd.to_numeric(df.loc[:, 'outcome'],
                                            errors='coerce')
        meanstdforoutcome = dict()
        # Because i standardize the outcome, i keep the values used for that
        # transformation (mean and std) so i can later transform back to the
        # original values (needed do later dicotomize the predicted outcome
        for treatment in ['Panic', 'Social_Anxiety', 'Depression']:
            meanstdforoutcome['{}'.format(treatment)] = (
                (pd.Series.mean(df.loc[df['Treat_org_drop']
                       == '{}'.format(treatment), 'outcome']),
                 pd.Series.std(df.loc[df['Treat_org_drop']
                       == '{}'.format(treatment), 'outcome']))
            )
            df.loc[df['Treat_org_drop'] == '{}'.format(treatment), 'outcome']=(
                StandardScaler().
                    fit_transform(
                    df.loc[df['Treat_org_drop'] ==
                           '{}'.format(treatment), 'outcome'].
                        values.reshape(-1, 1)))
        if not os.path.exists(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1','Handpicked')):
            print 'Result folder does not exist, creating results folder'
            os.makedirs(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1','Handpicked'))
        print 'Saving means and std for later for', meanstdforoutcome.keys()
        for name, values in meanstdforoutcome.iteritems():
            with open(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1','Handpicked',
                      '{}_{}_{}.txt'.format(date.strftime('%Y-%m-%d'),
                                        name,
                                        'meanstdforoutcome')), 'wb') as output:
                textout = ','.join([str(x) for x in values])
                output.write(textout.encode('utf8'))

    dfb = df.copy(deep=True) # temp

    # Standardising date columns
    df = convertdatescols(df)

    # merging all the main timepoints shared, and all the aspects of each
    # treatments mainsymptommeasure (the measurementitself - its duration -
    # its day during the week + its minute of the day
    print 'creating the mainsymptom features (measurement,duration,day,time)'
    for i in ['PRE', 'WEEK01', 'WEEK02', 'WEEK03']:
        for j in ['sum','duration','DateCompleted_day','DateCompleted_time']:
            df.loc[:,'mainsymptom_{}_{}'.format(i,j)] = np.nan
            df.loc[:, 'mainsymptom_{}_{}'.format(i, j)] = (
                df.apply(mainmerger, axis=1, args=[i,j]))
            todrop.extend(['PDSS-SR-3064_{}_{}'.format(i,j),
                       'MADRS-1951_{}_{}'.format(i,j),
                       'LSAS-2241_{}_{}'.format(i,j)])
    df.drop(todrop,axis='columns', inplace=True)
    return df, meanstdforoutcome



def splitintoweekwise(df):
    """ split the dataset into weekly dataset return a dictionary
    of datasets heavily reliant on names of columns and exr_clean_weekX"""
    dfdict = {}
    # -xx days
    names_screen = [x for x
                   in df.columns
                   if re.search(r'(outcome|1843|1833|currentwork|ncomorbid|'
                                r'^Treat.*|Depression|Panic|Social_Anxiety|'
                                r'sex|age$|'
                                r'SCREEN)', x)]

    dfdict['All_screen'] = df.loc[:,names_screen]
    # around day 0
    names_pre = [x for x
                   in df.columns
                   if re.search(r'(outcome|1843|1833|currentwork|ncomorbid|'
                                r'^Treat.*|Depression|Panic|Social_Anxiety|'
                                r'sex|age$|'
                                r'SCREEN|PRE)', x)]
    dfdict['All_pre'] = df.loc[:, names_pre]
    # around day 7
    # Due note that we do the first 7 days of treatment as week 1 (not 0)
    names_week01 = [x for x
                in df.columns
                if re.search(r'(outcome|1843|1833|currentwork|ncomorbid|'
                             r'^Treat.*|Depression|Panic|Social_Anxiety|'
                             r'sex|age$|'
                             r'SCREEN|PRE|^(messages|homeworks).*_7$)', x)]
    dfdict['All_week01'] = df.loc[:, names_week01]
    # around day 14
    names_week02 = [x for x
                    in df.columns
                    if re.search(r'(outcome|1843|1833|currentwork|ncomorbid|'
                                 r'^Treat.*|Depression|Panic|Social_Anxiety|'
                                 r'sex|age$|'
                                 r'SCREEN|PRE|.*WEEK01.*|^HW-01.*|'
                                 r'^(messages|homeworks).*_(7|14)$)', x)]
    dfdict['All_week02'] = exr_clean_weekX(df, names_week02, 13)
    # around day 21
    names_week03 = [x for x
                    in df.columns
                    if re.search(r'(outcome|1843|1833|currentwork|ncomorbid|'
                                 r'^Treat.*|Depression|Panic|Social_Anxiety|'
                                 r'sex|age$|'
                                 r'SCREEN|PRE|.*WEEK(01|02).*|^HW-0[1-2].*|'
                                 r'cscale|'
                                 r'^(messages|homeworks).*_(7|14|21)$)', x)]
    dfdict['All_week03'] = exr_clean_weekX(df, names_week03, 20)
    # Day 28 last
    names_week04 = [x for x
                    in df.columns
                    if re.search(r'(outcome|1843|1833|currentwork|ncomorbid|'
                                 r'^Treat.*|Depression|Panic|Social_Anxiety|'
                                 r'sex|age$|'
                                 r'SCREEN|PRE|.*WEEK(01|02|03).*|^HW-0[1-3].*|'
                                 r'cscale|'
                                 r'^(messages|homeworks).*_(7|14|21|28)$)', x)]
    dfdict['All_week04'] = exr_clean_weekX(df, names_week04, 28)
    return dfdict



def splitintotreatments(dfdict):
    """
    splits a dataframe dictionary into a larger dictionary also containing
    splits for each treatment
    :param dfdict: dictionary of dataframes (weekise splitted)
    :return: a bigger dictionary including dataframes spliited by treatment
    """
    print 'splitting into treatment specifics keeping features ' \
          'dropping'
    dfdictout = {}
    todrop = ['Treat_org_drop','Depression','Panic','Social_Anxiety',
              'HW-01_ChangedDayFromStart', 'HW-02_ChangedDayFromStart',
               'HW-03_ChangedDayFromStart']
    print todrop
    for name, df in dfdict.iteritems():
        for treatment in ['Depression', 'Panic', 'Social_Anxiety']:
            maskkeep = (
                    df.loc[df['Treat_org_drop'] == treatment, :].
                    isnull().mean() < 0.9) 
            dfsplit = df.loc[df['Treat_org_drop'] == treatment, maskkeep]
            todropspecific = [x for x in dfsplit.columns if x in todrop]
            dfsplit.drop(todropspecific, axis='columns',inplace=True)
            dfdictout['{}_{}'.format(treatment.split('_')[0],
                                     name.split('_')[-1])] = dfsplit
        # now drop the require columns from the all df too
        todropforalldf = [x for x in df.columns if 'ChangedDayFromStart' in x]
        todropforalldf.append('Treat_org_drop')
        dfdictout[name] = df.drop(todropforalldf, axis='columns')
    return dfdictout

def splitbenchmark(dfdict):
    """split dfdict for each measurementpoint to only have symptoms for
    baseline measurements"""
    dfdictout = {key: value[:] for key, value in dfdict.items()}
    for name, df in dfdict.iteritems():
        colsget = [x for x in list(df) if re.search(r'(outcome|'
                                                    r'(1951|3064|2241).*_sum$|'
                                                    r'sex|age$|'
                                                    r'^mainsymptom.*_sum$)',x)]
        dfdictout['{}_{}'.format(name,'benchmark')] = df.loc[:,colsget]
    return dfdictout



def missingremoval(dfdict,dfdicttrain, training=True):
    """
    removes all missing in the dataframe according to manual rules
    :param df: dataframe
    :return: dataframe cleaned from missing
    """
    print 'Handling missing'
    dictout = dict()
    for name, df in dfdict.iteritems():
        # Manual removing 2 non eligible gender variables
        df = df.loc[~df['sex'].isna(), :]

        colstofill = [x for x in df.columns if x in
                      ['HW-01', 'HW-02', 'HW-03',
                       'currentwork_proff',
                       'ncomorbid']]

        if training:
            # Mass removal
            maskkeep = (df.loc[:, :].isnull().mean() < 0.25) # 25 % missing 
            
            # when i have not already thrown out the no outcome people 
            print 'removing', len(list(set(df.columns)
                                       - set(df.loc[:, maskkeep].columns)))
            print '- columns b.c. > 25% miss for {}'.format(name)
            # double checking that outcome is kept 
            maskkeep.loc['outcome',] = True 
            
            dfnaremove = df.loc[:, maskkeep]

            # keeping a df with no imputation
            dfalmostnoimpute = dfnaremove.dropna(axis='rows')

            # now keep special columns for imputing
            maskkeep.loc[colstofill, ] = True

            df = df.loc[:, maskkeep]  # this now includes special columns


            imputer = MissForest(max_iter=10, n_jobs=-1,random_state=0)

            df.loc[:, :] = (
                imputer.fit_transform(df)
            )
            if not os.path.exists(os.path.join(PATH, '..',
                                                      'datasets', 'processed',
                                                      'study1',
                                                      'Handpicked',
                                               'missimputeobj')):
                print 'Missimputeobj folder does not exist, creating folder'
                os.makedirs(os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1',
                                  'Handpicked',
                                  'missimputeobj'))
            try:
                dump(imputer,
                     os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1',
                                  'Handpicked',
                                  'missimputeobj',
                         '{}_{}_{}.joblib'.format(date.strftime('%Y-%m-%d'),
                                                  name, 'imputer')))
            except NameError:
                pass
        else:
            # Mass removal by colname from training
            # this is done in 2 versions b.c. different for
            # imputing and naremoval
            # Na removal first

            # keeping a df with no imputation
            try:
                if 'benchmark' in name.split('_'):
                    namingna = '{}-{}_{}'.format('_'.join(name.split('_')[0:2]),
                                                'naremove',
                                                '_'.join(name.split('_')[2:]))
                else:
                    namingna = '{}-naremove'.format(name)
                maskkeepna = dfdicttrain[namingna].columns
            except KeyError as ek:
                print 'keys are',dfdicttrain.keys()
                print ek, 'occured'
                quit()

            dfnaremove = df.loc[:, maskkeepna]

            if len(dfnaremove.columns) != len(maskkeepna):
                print 'for ', name
                print 'error', 'train and test not same columns'
                print 'testshape', df.shape
                print 'trainshape', dfdicttrain[namingna].shape
                quit()

            dfalmostnoimpute = dfnaremove.dropna(axis='rows')

            # imputing
            try:
                if 'benchmark' in name.split('_'):
                    naming1 = '{}-{}_{}'.format('_'.join(name.split('_')[0:2]),
                                                'imputed',
                                                '_'.join(name.split('_')[2:]))
                else:
                    naming1 = '{}-imputed'.format(name)
                maskkeep = dfdicttrain[naming1].columns
            except KeyError as ek:
                print 'keys are',dfdicttrain.keys()
                print ek, 'occured'
                quit()

            df = df.loc[:, maskkeep]

            if len(df.columns) != len(maskkeep):
                print 'for ', name
                print 'error', 'train and test not same columns'
                print 'testshape', df.shape
                print 'trainshape', dfdicttrain[naming1].shape
                quit()

            imputerfile = newest(os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1',
                                  'Handpicked',
                                  'missimputeobj'),
                                 '.*{}_imputer'.format(name))

            imputer = load('{}'.format(imputerfile))  # saved below
            imputer.missing_values = np.nan

            try:
                df.loc[:, :] = (
                    imputer.transform(df)
                )
            except Exception as tenoo:
                print name
                print df
                print tenoo, 'Warning exception quitting '
                quit()
        if 'benchmark' in name.split('_'):
            naming1 = '{}-{}_{}'.format('_'.join(name.split('_')[0:2]),
                                        'imputed',
                                        '_'.join(name.split('_')[2:]))
            naming2 = '{}-{}_{}'.format('_'.join(name.split('_')[0:2]),
                                        'naremove',
                                        '_'.join(name.split('_')[2:]))
        else:
            naming1 = name + '-imputed'
            naming2 = name + '-naremove'
        dictout['{}'.format(naming1)] = df
        dictout['{}'.format(naming2)] = dfalmostnoimpute
    return dictout

def wrapper(outcome,arbetsblad,instruments,whatoutcome=outcomedict):
    """
    wrapper function fusing together the different datasources and returning
    a dictionary of all dataframes needed
    :param outcome:  df of outcome
    :param arbetsblad: df of arbetsblad (widef ormat)
    :param instruments: df of instruments
    :return:dictionary of dataframes
    """
    # importing all functions and cleaning up each individual datasource
    # dflabeloutcome contains information about the patient and their outcome
    # so i can later use this for manual scoring functions
    df, dflabeloutcome = mergedfssetupfeatures(outcome,arbetsblad,instruments,
                               whatoutcome=whatoutcome)

    # merging symptoms and standardising columns
    print 'Merging symptoms and standardising columns'
    # meanstdforoutcome contains the values needed to transform continous
    # outcome back to original scale
    df, meanstdforoutcome = merged_cleansymptoms(df)
    dfb2 = df.copy(deep=True)

    # Remove redudant columns
    removecols = [x for x in df.columns
                   if re.search(r'(POST|^TreatmentAccess.*)',x)]
    df.drop(removecols,axis='columns',inplace=True)
    if CONTOUTCOME:
        pass
    else:
        exit('dict outcome outdated')
        df['outcome'] = df['outcome'].replace({'Success': 1, 'Failure': 0})

    # randomise order on indviduals before split
    df = df.sample(frac=1)

    # Make sure dtypes are correct here
    df['currentwork_proff'] = pd.to_numeric(df['currentwork_proff'],
                                            errors='coerce')
    df.loc[:,df.dtypes=='uint8'] = df.loc[:,df.dtypes=='uint8'].astype(int)

    # Some manual removal because too much missing
    colstoremove = ['HW-01-2_ChangedDayFromStart', 'HW-01-3_ChangedDayFromStart',
                    'HW-01-4_ChangedDayFromStart', 'HW-01-2', 'HW-01-3', 'HW-01-4']
    df = df.loc[:, [x for x in df.columns if x not in colstoremove]]

    dfb3 = df.copy(deep=True)

    print 'Done!'

    print 'Starting to chop up the dataset here'
    # split the data into weekly-wise datasets
    # splitting missing first
    #dfmissd = {}

    #dfmissd['_imputed'] = df
    #dfmissd['_naremove'] = dfnar

    dfdict_week = splitintoweekwise(df) 

    # split the data once again by treatment + all groupation
    dfdict_week_treat = splitintotreatments(dfdict_week)

    dfdict_week_treat_bench = splitbenchmark(dfdict_week_treat)

    print 'Saving all dataframes in ', os.path.join(PATH, '..',
                                                    'datasets',
                                                    'processed', 'study1',
                                                    'Handpicked')
    if not os.path.exists(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1',
                                       'Handpicked', 'Alldata')):
        print 'Result folder does not exist, creating results folder'
        os.makedirs(os.path.join(PATH, '..',
                                 'datasets', 'processed', 'study1',
                                 'Handpicked', 'Alldata'))
    savedfs(dfdict_week_treat_bench, os.path.join(PATH, '..',
                                           'datasets', 'processed', 'study1',
                                           'Handpicked', 'Alldata'),
            date=date.strftime('%Y-%m-%d'),
            in_production=False)

    print 'creating split for test and train' #holdout 
    smallerdict = {}
    for name, df in dfdict_week_treat_bench.iteritems():
        smallerdict['{}'.format(name)] = df.sample(frac=0.1, random_state=0)
        dfdict_week_treat_bench[name] = (
            df.drop(smallerdict[name].index)
        )

    # split the data once again creating another dataset for each dataset
    # containing only the primary symptoms
    print 'more saving test/train'
    if not os.path.exists(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1',
                                       'Handpicked', 'train')):
        os.makedirs(os.path.join(PATH, '..',
                                 'datasets', 'processed', 'study1',
                                 'Handpicked', 'train'))
    savedfs(dfdict_week_treat_bench, os.path.join(PATH, '..',
                                           'datasets', 'processed', 'study1',
                                           'Handpicked', 'train'),
            date=date.strftime('%Y-%m-%d'),
            in_production=False)
    # and test
    if not os.path.exists(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1',
                                       'Handpicked', 'test')):
        os.makedirs(os.path.join(PATH, '..',
                                 'datasets', 'processed', 'study1',
                                 'Handpicked', 'test'))
    savedfs(smallerdict, os.path.join(PATH, '..',
                                      'datasets', 'processed', 'study1',
                                      'Handpicked', 'test'),
            date=date.strftime('%Y-%m-%d'),
            in_production=False)

    dfdictmissfixtrain = reloaddfs(regexfile=r'.*_.*_.*.csv',
                              fullPATH=os.path.join(PATH, '..',
                                                    'datasets', 'processed',
                                                    'study1',
                                                    'Handpicked',
                                                    'train'))
    dfdictmissfixfixedtrain = missingremoval(dfdictmissfixtrain,
                                            dfdicttrain={},
                                            training=True)

    if not os.path.exists(os.path.join(PATH, '..',
                                       'datasets',
                                       'processed',
                                       'study1',
                                       'Handpicked',
                                       'train',
                                       'missfix')):
        os.makedirs(os.path.join(PATH, '..',
                                 'datasets', 'processed', 'study1',
                                 'Handpicked',
                                 'train',
                                 'missfix'))
    savedfs(dfdictmissfixfixedtrain, os.path.join(PATH, '..',
                                      'datasets', 'processed', 'study1',
                                      'Handpicked',
                                             'train',
                                             'missfix'),
            date=date.strftime('%Y-%m-%d'),
            in_production=False)

    # d names are now 'Treatment_timepoint-nahandling_benchmarkornobenchmark'
    # ex. Depression_screen_imputed, Depression_screen-imputed_benchmark

    dfdictmissfixtest = reloaddfs(regexfile=r'.*_.*_.*.csv',
                              fullPATH=os.path.join(PATH, '..',
                                                    'datasets', 'processed',
                                                    'study1',
                                                    'Handpicked',
                                                    'test'))
    dfdictmissfixfixedtest = missingremoval(dfdictmissfixtest,
                                            dfdicttrain=dfdictmissfixfixedtrain,
                                            training=False)
    if not os.path.exists(os.path.join(PATH, '..',
                                       'datasets',
                                       'processed',
                                       'study1',
                                       'Handpicked',
                                       'test',
                                       'missfix')):
        os.makedirs(os.path.join(PATH, '..',
                                 'datasets', 'processed', 'study1',
                                 'Handpicked',
                                 'test',
                                 'missfix'))
    savedfs(dfdictmissfixfixedtest, os.path.join(PATH, '..',
                                             'datasets', 'processed', 'study1',
                                             'Handpicked',
                                             'test',
                                             'missfix'),
            date=date.strftime('%Y-%m-%d'),
            in_production=False)
    print 'Done chopping!'

    return dfdictmissfixfixedtrain, meanstdforoutcome, dflabeloutcome

##############################################################################
# analyses
##############################################################################


def my_score_conttodict(y_true, y_pred, X_test,
                        dflabeloutcome,meanstdforoutcome):
    #df here is really X from predictionn
    """
    Makes custom scoring functions
    :param y_true: the true outcome value (vector)
    :param y_pred: the predicted outcome value (vector)
    :param X_test: the dataframe which we used to predict the outcome with
    :param dflabeloutcome: dataframe containing all information needed to
    label a patient from a continous prediction (patient info, the pre-treatment
    main symptom score, the treatment patient was in, the dicotomized score)
    :param meanstdforoutcome: the mean+std for each continous outcome for each
    of the 3 treatments (needed to transform the prediciton back to
    the original scale)
    :return: a dictionary of scoring function how well the true outcome
    compares to the predicted outcome
    """
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(0, len(y_true)):
        # get patient (from the dataset used for prediction)
        patient = X_test.iloc[i].name

        # get treatment for patient - utilizing the previous saved dataframe
        # containing the information needed to find the patient and the scores
        treatment = (
            dflabeloutcome.loc[
                dflabeloutcome.index == patient, 'Treat_org_drop'].values[0])
        try:
            score_m, score_sd = meanstdforoutcome[treatment]
        except KeyError:
            # guard against bad naming convention of social_anxiety treatment
            score_m, score_sd = [meanstdforoutcome[x] for x in meanstdforoutcome if
                      re.search(r'{}'.format(treatment.split('_')[0]),x)][0]

        # get unscaled scores for
        t_score = float(y_true.item(i)*score_sd+score_m)
        p_score = float(y_pred.item(i)*score_sd+score_m)
        # get true starting score
        s_score = float(dflabeloutcome.loc[dflabeloutcome.index == patient,
                                     'mainsymptom_PRE_sum'].values[0])
        if treatment == 'Panic' and s_score is np.nan:
            s_score = float(dflabeloutcome.loc[
                dflabeloutcome.index==patient,'PDSS-SR-3064_SCREEN_sum'].values[0])
        if treatment == 'Social_Anxiety' and s_score is np.nan:
            s_score = float(dflabeloutcome.loc[
                dflabeloutcome.index == patient, 'LSAS-2241_SCREEN_sum'].values[0])
        if treatment == 'Depression' and s_score is np.nan:
            s_score = float(dflabeloutcome.loc[
                dflabeloutcome.index == patient, 'MADRS-1951_SCREEN_sum'].values[0])

        # get limit
        limit = LIMIT[treatment]

        # Get true label
        # The true label is imputed (if unknown from the start)
        #  and HAVE to be calculated from y_true and s_score 
        if (t_score <= limit) or (t_score <=(s_score*0.5)): 
            t_label = 'Success'
        else: 
            t_label = 'Failure'

        #t_label = (
        #    dflabeloutcome.loc[dflabeloutcome.index == patient, 'outcomesave'].
        #    values[0])

        # Get label for predicted score
        if (p_score <= limit) or (p_score <= (.5*s_score)):
            p_label = 'Success'
        else:
            p_label = 'Failure'
        if t_label == 'Success':
            if p_label == 'Success':
                TP += 1
            if p_label == 'Failure':
                FN += 1
        if t_label == 'Failure':
            if p_label == 'Success':
                FP += 1
            if p_label == 'Failure':
                TN += 1
    total = TP+TN+FP+FN
    try:
        precision = float(TP)/float(TP+FP)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = float(TP)/float(TP+FN)
    except ZeroDivisionError:
        recall = 0
    try:
        specificity = float(TN)/float(FP+TN)
    except ZeroDivisionError:
        specificity = 0
    try:
        b_acc = (recall + specificity)/float(2)
    except ZeroDivisionError:
        b_acc = 0
    try:
        f1 = (1+np.power(1,2))*(precision*recall)/(np.power(1,2)*precision+recall)
    except ZeroDivisionError:
        f1 = 0
    dictout = {'precision':precision,'recall':recall,'specificity':specificity,
               'b-acc':b_acc,'f1':f1,'TP':TP,'FP':FP,'FN':FN,'TN':TN}
    return dictout

def my_score_bin(y_true, y_pred):

    """
    """
    confusionmatrix = confusion_matrix(y_true, y_pred)
    TP = confusionmatrix[1,1]
    FN = confusionmatrix[1,0]
    FP = confusionmatrix[0,1]
    TN = confusionmatrix[0,0]
    total = TP+TN+FP+FN
    try:
        precision = float(TP)/float(TP+FP)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = float(TP)/float(TP+FN)
    except ZeroDivisionError:
        recall = 0
    try:
        specificity = float(TN)/float(FP+TN)
    except ZeroDivisionError:
        specificity = 0
    try:
        b_acc = (recall + specificity)/float(2)
    except ZeroDivisionError:
        b_acc = 0
    try:
        f1 = (1+np.power(1,2))*(precision*recall)/(np.power(1,2)*precision+recall)
    except ZeroDivisionError:
        f1 = 0
    dictout = {'precision':precision,'recall':recall,'specificity':specificity,
               'b-acc':b_acc,'f1':f1,'TP':TP,'FP':FP,'FN':FN,'TN':TN}
    return dictout

def my_kf_model_builder(X,
                        y,
                        dataset,
                        name,  # name classifier
                        outcome,
                        classifier,
                        results,
                        dflabeloutcome,
                        meanstdforoutcome,
                        Xtest=None,
                        ytest=None,
                        PCAUSE=False,
                        pca=False):
    if not PCAUSE:
        if 'benchmark' in dataset.split('_'):
            PCAUSE = 'Benchmark'
        else:
            PCAUSE = 'Handpicked'
    appendresults = pd.DataFrame({'Treatment': '{}'.
                                 format(dataset.split('_')[0]),
                                  'Outcome': '{}'.
                                 format(outcome),
                                  'Time': '{}'.
                                 format(dataset.split('_')[1].split('-')[0]),
                                  'Data_amount': '{}'.
                                 format(dataset.split('_')[1].split('-')[1]),
                                  'Data_shape': '{}'.
                                 format(X.shape),
                                  'PCA': '{}'.
                                 format(str(PCAUSE)),
                                  'PCA_params': '{}'.
                                 format(str(pca)),
                                  'Method': '{}'.
                                 format(name),
                                  'Params': '{}'.
                                 format(classifier),
                                  }, index=[0])
    # k fold cross validation using the scores designated above
    # return estimator returns amongst other things feature_importance
    kf = KFold(n_splits=10, random_state=0)
    myscorers = dict()
    mykfscores = pd.DataFrame(columns=['r2', 'mse', 'mae', 'precision',
                                       'recall', 'specificity',
                                       'b-acc', 'f1'])
    mykfscores_ho = pd.DataFrame(columns=['r2_ho', 'mse_ho', 'mae_ho',
                                          'precision_ho', 'recall_ho',
                                          'specificity_ho', 'b-acc_ho', 'f1_ho'])
    # holdout test
    if Xtest is not None:
        print 'Holdout prediction'
        fit = classifier  # get classifier
        fit.fit(X, y.ravel())  # fit using original data (90%)
        ytestpred = fit.predict(Xtest)
        ytesttrue = ytest
        if outcome == 'symptom-cont':
            myscorers_ho_raw = my_score_conttodict(ytesttrue, ytestpred, Xtest,  # X_test gets patient name
                                            dflabeloutcome,
                                            meanstdforoutcome)
            myscorers_ho_raw['r2'] = r2_score(ytesttrue, ytestpred)
            myscorers_ho_raw['mse'] = mean_squared_error(ytesttrue, ytestpred)
            myscorers_ho_raw['mae'] = mean_absolute_error(ytesttrue, ytestpred)
        else:
            myscorers_ho_raw = {}  # placeholder
        myscorers_ho = {k+'_ho': v for k, v in myscorers_ho_raw.iteritems()}
        mykfscores_ho = mykfscores_ho.append(myscorers_ho, ignore_index=True)

    print 'Training with {}, K-fold CV with K = {}'.format(name, kf.n_splits)
    for train_row, test_row in kf.split(X):
        # print("TRAIN:", train_row, "TEST:", test_row)
        # fit the classifier
        fit = classifier
        X_train, X_test = X.iloc[train_row], X.iloc[test_row]
        y_train, y_test = y[train_row], y[test_row]
        fit.fit(X_train, y_train.ravel())
        y_pred = fit.predict(X_test)
        y_true = y_test
        if outcome == 'symptom-cont':
            myscorers = my_score_conttodict(y_true, y_pred, X_test,  # X_test gets patient name
                                            dflabeloutcome,
                                            meanstdforoutcome)
            myscorers['r2'] = r2_score(y_true, y_pred)
            myscorers['mse'] = mean_squared_error(y_true, y_pred)
            myscorers['mae'] = mean_absolute_error(y_true, y_pred)
        elif outcome in labels_bin:
            myscorers = my_score_bin(y_true, y_pred)
        elif outcome in labels_cont:
            myscorers['r2'] = r2_score(y_true, y_pred)
            myscorers['mse'] = mean_squared_error(y_true, y_pred)
            myscorers['mae'] = mean_absolute_error(y_true, y_pred)
        mykfscores = mykfscores.append(myscorers, ignore_index=True)
        try:
            featimparray = fit.feature_importances_
            featimparraynames = ['feat-imp_' + x for x in X.columns]
            featimpdict = dict(zip(featimparraynames, featimparray))
            mykfscores = mykfscores.append(featimpdict, ignore_index=True)
        except AttributeError as error:
            # print 'classifier', name, 'has no featureimp'
            pass

    # Take the mean of the k values as accuracy
    for test_name, col_values in mykfscores.iteritems():
        appendresults[test_name] = np.mean(col_values)
        appendresults[test_name + '_std'] = np.std(col_values)
    for test_name,value in mykfscores_ho.iteritems():
        appendresults[test_name] = value
    # meanstdforoutcome <- dictionary for all meanstds
    results = results.append(appendresults, sort=True)  # append the results
    return results

def my_model_buildsaver(X,
                        y,
                        path,
                        dataset,
                        name, # name classifier
                        module_name,
                        classifier):
    # fit object
    try:
        date
    except Exception:
        date = datetime.datetime.today().strftime('%Y-%m-%d')
    print 'Fitting', name, 'to', dataset
    fit = classifier
    fit.fit(X, y.ravel())
    if not os.path.exists(os.path.join(path, '..', 'models')):
        print 'Models folder does not exist, creating folder'
        os.makedirs(os.path.join(path, '..', 'models'))
    dump(fit,
         os.path.join(
             path,
             '..',
             'models',
             '{}_{}_{}_{}.joblib'.format(date, module_name, dataset, name)))
    # save object
    # return nothing
    return fit


def runanalyses(dfd,
                PATH,
                date,
                dflabeloutcome,
                meanstdforoutcome,
                hyperparams,
                classifiers_by_label,
                in_production,
                diagnostics,
                modelbuilding,
                dfdtest=False,
                PCAUSE=False,
                pcaoptions=False,
                anymiss=False,
                savedf=False,
                outcome=False,
                module_name = False):
    """
    The analyses function
    :param dfd: a dictionary of dataframes to be analysed 
    :param dflabeloutcome: dataframe containing all information needed to
    label a patient from a continous prediction (patient info, the pre-treatment
    main symptom score, the treatment patient was in, the dicotomized score)
    :param meanstdforoutcome: the mean+std for each continous outcome for each
    of the 3 treatments (needed to transform the prediciton back to
    the original scale)
    :param scores: the scores to evalute the models on (dict)
    :param anymiss: is there any missings in the dataset?
    :param savedf: do we want to save all the dataframes to folder?
    :return: return nothing just run analyses and save ouput in csv
    """
    if not os.path.exists(os.path.join(PATH,'..','results')):
        print 'Result folder does not exist, creating results folder'
        os.makedirs(os.path.join(PATH,'..','results'))
    if not os.path.exists(os.path.join(PATH,'..','results','production')):
        print 'Results/production folder does not exist, creating results folder'
        os.makedirs(os.path.join(PATH,'..','results','production'))

    if savedf:
        if PCAUSE != False:
            print ('NOTE ------------------ \n ',
                  'PCA is used dataframes will be transformed but '
                  'transformation is not saved \n ',
                  '----------------- \n')
        print 'Saving all dataframes in ', os.path.join(PATH,'..','datasets', 'processed')
        if not os.path.exists(os.path.join(PATH,'..','datasets', 'processed')):
            print 'Folder does not exist, creating datasets processed folder'
            os.makedirs(os.path.join(PATH,'..','datasets', 'processed'))
        savedfs(dfd, os.path.join(PATH,'..','datasets', 'processed'),
                whatoutcome=outcome)

    # Now go through teach dataset and fit classifiers + results to it
    if diagnostics:
        print 'Running and creating diagnostics'
        # sort and get all the diagnostic scores
        colnames = sorted(['precision', 'recall', 'specificity',
                           'b-acc', 'f1', 'TP', 'FP', 'FN', 'TN', 'r2', 'mse', 'msa'])

        # add the dataset and method to the results
        colnames[0:0] = ['Treatment', 'Outcome', 'Time',
                         'Method', 'Params', 'PCA_params']
        # create empty dataframe
        results = pd.DataFrame(columns=colnames)
        for dataset, df in tqdm(dfd.iteritems(), total=len(dfd)):

            if dfdtest:
                dftest = dfdtest[dataset]
                # Splitting into features and outcome
                Xtest_org, ytest_org = (dftest.loc[:, dftest.columns != 'outcome'],
                            dftest.loc[:, dftest.columns == 'outcome'].values)

            print '\n Processing {}'.format(dataset)

            #'Treatment_timepoint-nahandling_benchmarkornobenchmark'

            # dataset = treatment_label_week01_imputed

            if dataset.endswith('benchmark'):
                dataname = dataset.split('_')[0] + '_' + dataset.split('_')[-1]
            else:
                dataname = dataset.split('_')[0]

            # check if this is right
                # check if this is right
            if classifiers_by_label:
                outcome = [part for part in dataset.split('_')
                           if part in labeldictionary.keys()][0]
                classifiers = classifiers_by_label[outcome]
            elif outcome:
                from modules.hyperparamtuning import classifiers_by_label as clf_label
                classifiers = clf_label[outcome]
            else:
                classifiers = classifiersIDconthyper  # list with instances

            # Splitting into features and outcome
            X_org, y = (df.loc[:, df.columns != 'outcome'],
                        df.loc[:, df.columns == 'outcome'].values)

            # based on dataset get 'true dict values' to use in custome make scorer
            # y_dict_true = df.loc[
            if PCAUSE:
                if PCAUSE == 'linear':  # must align with the passed along hyperparams
                    pca = PCA(**pcaoptions)
                    #pcaobj = newest(os.path.join(PATH, '..',
                    #                   'datasets', 'processed',
                    #                         'study1', 'PCA'),
                    #    '.*{}_{}$'.format(dataset, 'pca_transform_linear.joblib'))
                    #
                    #pca = load('{}'.format(pcaobj))
                    Xpca = pd.DataFrame(data=pca.fit_transform(X_org))
                    Xpca.columns = ['pc' + str(x) for x in Xpca.columns]
                    Xpca.index = X_org.index
                    X = Xpca
                    writefile(X,
                              '{}_{}'.format(dataset,'linearPCA'),
                              date=date,
                              fullPATH = os.path.join(
                                  PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'PCA', 'train', 'missfix', 'transformed')
                              )

                    #test data too
                    Xtest = pd.DataFrame(data=pca.transform(Xtest_org))
                    Xtest.columns = ['pc' + str(x) for x in Xtest.columns]
                    Xtest.index = Xtest_org.index
                    writefile(Xtest,
                              '{}_{}'.format(dataset, 'linearPCA'),
                              date=date,
                              fullPATH=os.path.join(
                                  PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'PCA', 'test', 'missfix', 'transformed')
                              )
                    ytest = ytest_org

                elif PCAUSE == 'kernel':
                    pass
                else:
                    sys.exit("PCA not properly defined")
            else:
                # undue fiddling with variables due to compatibale with 3 different
                # data manipulations (no pca, linear pca, kernel pca)
                print 'No PCA using original given DataFrame(s)'
                X = X_org
                Xtest = Xtest_org
                ytest = ytest_org
            for classifier in classifiers:
                try:
                    # get name of quantile regression o.w. just get
                    # classifier name
                    if classifier.get_params()['loss'] == 'quantile':
                        name = '{}_{}'.format(
                            type(classifier).__name__,
                            ''.join(
                                str(classifier.get_params()['alpha']).split('.')
                            ))
                        continue  # if you have a quantile loss then skip
                    else:
                        name = type(classifier).__name__
                except KeyError:
                    name = type(classifier).__name__

                if hyperparams is not False:  # Try load the hyperparameters
                    paramsforclassifier = hyperparams.loc[
                        (hyperparams['dataset'] == dataset) &
                        (hyperparams['classifier'] == name), 'bestparams']

                    get_params_from_regex = False

                    if ((len(paramsforclassifier) == 0) &
                            (name in set(hyperparams['classifier']))):
                        try:  # Try regex load hyperparameters
                            #print('regex matching to get params')
                            get_params_from_regex = True
                            hyperparamdfname = '.*'.join(
                                [dataset.split('_')[0],
                                 dataset.split('-')[1]])
                            # Now get the params for the classifier
                            paramsforclassifier = hyperparams.loc[
                                (hyperparams['dataset'].str.contains('{}{}'.format(hyperparamdfname, '$'))) &
                                (hyperparams['classifier'] == name), 'bestparams']
                        except Exception as t:
                            # there are no hyperparameters
                            print 'no hyperparams, regex failed '
                            paramsforclassifier = False
                        if len(paramsforclassifier) == 0:
                            print 'no hyperparams, regex could not load properly'
                            paramsforclassifier = False
                    else:
                        pass
                        #print 'Uses params from datasettuning'
                        #paramsforclassifier = False
                else:
                    print 'no hyperparams, params are false'
                    paramsforclassifier = False

                # get sklearn name for classifier
                # Get the name of the dataset in regex format so we can match against
                #  the hyperparamtune name of dataset or closest regex

                if PCAUSE == 'kernel':
                    # Get params for kernel from tuning
                    bestpcaoptions = hyperparams.loc[
                        (hyperparams['dataset'].str.contains('{}{}'.format(hyperparamdfname, '$'))) &
                        (hyperparams['classifier'] == name), 'pcaparams']
                    try:
                        bestpcaoptionsdict = ast.literal_eval(
                            bestpcaoptions.ravel()[0])
                    except ValueError as e:
                        # This happens for model which initiates other models
                        #  such as Adaboost which param dictionary initiates a RF model
                        bestpcaoptionsdict = eval(
                            bestpcaoptions.ravel()[0])
                    # TODO POSSIBLE SOLUTION
                    #  add caveat if u get allof of missing because more dimensions than features
                    # because hypertuning is done on a larger dataset
                    #pcaobjk = newest(os.path.join(PATH, '..',
                    #                             'datasets', 'processed',
                    #                             'study1', 'PCA'),
                    #                '.*{}_{}_{}$'.format(dataset,classifier,
                    #                                  'pca_transform_kernel.joblib'))
                    #pca = load('{}'.format(pcaobjk))
                    pca = KernelPCA(**bestpcaoptionsdict)
                    try:
                        Xpca = pd.DataFrame(data=pca.fit_transform(X_org))
                    except RuntimeWarning:
                        # if lower dataset is not optimised as week4 is
                        pca.set_params(**{'n_components': None})
                        Xpca = pd.DataFrame(data=pca.fit_transform(X_org))
                    Xpca.columns = ['pc' + str(x) for x in Xpca.columns]
                    Xpca.index = X_org.index
                    X = Xpca
                    pcaoptions = pca.get_params()
                    writefile(X,
                              '{}_{}'.format(dataset, 'kernelPCA'),
                              date=date,
                              fullPATH=os.path.join(
                                  PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'PCA', 'train', 'missfix', 'transformed')
                              )

                    Xtest = pd.DataFrame(data=pca.transform(Xtest_org))
                    Xtest.columns = ['pc' + str(x) for x in Xtest.columns]
                    Xtest.index = Xtest_org.index
                    writefile(Xtest,
                              '{}_{}'.format(dataset, 'kernelPCA'),
                              date=date,
                              fullPATH=os.path.join(
                                  PATH, '..',
                                  'datasets', 'processed',
                                  'study1', 'PCA', 'test', 'missfix', 'transformed')
                              )
                    ytest = ytest_org

                if paramsforclassifier is not False:
                    try:
                        print 'Unraveling params for classifier', paramsforclassifier
                        paramforclassifierdict = ast.literal_eval(
                            paramsforclassifier.ravel()[0])
                    except ValueError as e:
                        # This happens for model which initiates other models
                        #  such as Adaboost which param dictionary initiates a RF model
                        paramforclassifierdict = eval(
                            paramsforclassifier.ravel()[0])
                    except IndexError:
                        print dataset,hyperparamdfname,paramsforclassifier
                    # now set the classifier parameters from hyperparam tuning
                    classifier.set_params(**paramforclassifierdict)
                else:
                    pass

                # Diagnose purpose
                if paramsforclassifier is not False:
                    if get_params_from_regex:
                        print 'For', dataset, '\n and ', name, '\n using params for \n ', \
                            hyperparams.loc[
                                (hyperparams['dataset'].str.contains('{}{}'.format(hyperparamdfname, '$'))) &
                                (hyperparams['classifier'] == name), ['dataset', 'classifier']]
                    else:
                        print 'For', dataset, '\n and ', name, '\n using params for \n ', \
                            hyperparams.loc[
                                (hyperparams['dataset'] == dataset) &
                                (hyperparams['classifier'] == name), ['dataset', 'classifier']]
                else:
                    pass
                # append the name of current dataset and the classifier
                # pick only name (first) then timepoint then dataamount
                results = my_kf_model_builder(X,
                                              y,
                                              dataset,
                                              name,
                                              outcome,
                                              classifier,
                                              results,
                                              dflabeloutcome,
                                              meanstdforoutcome,
                                              Xtest=Xtest,
                                              ytest=ytest,
                                              PCAUSE=PCAUSE,
                                              pca=pcaoptions)
        if anymiss:
            missing = 'missingdf'
        else:
            missing = 'nomiss'
        print '\n saving results to', os.path.join(PATH, '..', 'results',
                                                   '{}_{}_ML_results_{}.csv'.
                                                   format(date,
                                                          module_name,
                                                          'pca-' + str(PCAUSE)))
        with open(os.path.join(PATH, '..', 'results', '{}_{}_ML_results_{}.csv'.
                format(date,
                       module_name,
                       'pca-' + str(PCAUSE))), 'wb') as output:
            results.to_csv(path_or_buf=output,
                           encoding='utf8',
                           index=False,
                           mode='wb')
        print '\n saving top results to', \
            os.path.join(PATH, '..', 'results',
                         '{}_{}_ML_results_{}_{}_maxed.csv'.
                         format(date, module_name, missing, 'pca-' + str(PCAUSE)))
        resultsmax = (results.groupby(['Treatment', 'Time', 'Outcome'])['b-acc'].
                      transform(max) == results['b-acc'])
        dfmaxresults = results[resultsmax]
        dfmaxresults = dfmaxresults.sort_values(by=['Time'])
        with open(os.path.join(PATH, '..', 'results', '{}_{}_ML_results_{}_maxed.csv'.
                format(date,
                       module_name, 'pca-' + str(PCAUSE))), 'wb') as output:
            dfmaxresults.to_csv(path_or_buf=output,
                                encoding='utf8',
                                index=False,
                                mode='wb')
        print ' \n Done! '
    if modelbuilding:
        print 'Building models'
        for dataset, df in tqdm(dfd.iteritems(), total=len(dfd)):
            print '\n Processing {}'.format(dataset)
            # 2020-01-02_(LIVE|UPDATE)_treatment_label_week01_imputed
            # dataset = treatment_label_week01_imputed

            # Dataname is only used to get the classifiers so it is now deprecated
            # Could be redudant now in 20200129
            if dataset.endswith('benchmark'):
                dataname = dataset.split('_')[0] + '_' + dataset.split('_')[-1]
            else:
                dataname = dataset.split('_')[0]

            # check if this is right
            if classifiers_by_label:
                outcome = [part for part in dataset.split('_')
                           if part in labeldictionary.keys()][0]
                classifiers = classifiers_by_label[outcome]
            else:
                classifiers = classifiersIDconthyper  # list with instances

            # Splitting into features and outcome
            X_org, y = (df.loc[:, df.columns != 'outcome'],
                        df.loc[:, df.columns == 'outcome'].values)
            # based on dataset get 'true dict values' to use in custome make scorer
            # y_dict_true = df.loc[
            if PCAUSE:
                if PCAUSE == 'linear':  # must align with the passed along hyperparams
                    pca = PCA(**pcaoptions)
                    Xpca = pd.DataFrame(data=pca.fit_transform(X_org))
                    Xpca.columns = ['pc' + str(x) for x in Xpca.columns]
                    Xpca.index = X_org.index
                    X = Xpca
                elif PCAUSE == 'kernel':
                    pass
                else:
                    sys.exit("PCA not properly defined")
            else:
                # undue fiddling with variables due to compatibale with 3 different
                # data manipulations (no pca, linear pca, kernel pca)
                print 'No PCA using original given DataFrame(s)'
                X = X_org
            for classifier in classifiers:

                try:
                    # get name of quantile regression o.w. just get
                    # classifier name
                    if classifier.get_params()['loss'] == 'quantile':
                        name = '{}_{}'.format(
                            type(classifier).__name__,
                            ''.join(
                                str(classifier.get_params()['alpha']).split('.')
                            ))
                    else:
                        name = type(classifier).__name__
                except KeyError:
                    name = type(classifier).__name__

                if hyperparams is not False:  # Try load the hyperparameters
                    # TODO for the love of ... clean up this check
                    paramsforclassifier = hyperparams.loc[
                        (hyperparams['dataset'] == dataset) &
                        (hyperparams['classifier'] == name), 'bestparams']

                    get_params_from_regex = False
                    if ((len(paramsforclassifier) == 0) &
                            (name in set(hyperparams['classifier']))):
                        try:  # Try regex load hyperparameters
                            get_params_from_regex = True
                            hyperparamdfname = '.*'.join(
                                ['_'.join(dataset.split('_')[0:2]),
                                 dataset.split('_')[-1]])
                            # Now get the params for the classifier
                            paramsforclassifier = hyperparams.loc[
                                (hyperparams['dataset'].str.contains(hyperparamdfname)) &
                                (hyperparams['classifier'] == name), 'bestparams']
                        except Exception as t:
                            # there are no hyperparameters
                            print 'no hyperparams'
                            paramsforclassifier = False
                        if len(paramsforclassifier) == 0:
                            print 'no hyperparams'
                            paramsforclassifier = False
                    else:
                        print 'no hyperparams'
                        paramsforclassifier = False
                else:
                    print 'no hyperparams'
                    paramsforclassifier = False

                # get sklearn name for classifier
                # Get the name of the dataset in regex format so we can match against
                #  the hyperparamtune name of dataset or closest regex

                if PCAUSE == 'kernel':
                    # Get params for kernel from tuning
                    bestpcaoptions = hyperparams.loc[
                        (hyperparams['dataset'].str.contains(hyperparamdfname)) &
                        (hyperparams['classifier'] == name), 'pcaparams']
                    try:
                        bestpcaoptionsdict = ast.literal_eval(
                            bestpcaoptions.ravel()[0])
                    except ValueError as e:
                        # This happens for model which initiates other models
                        #  such as Adaboost which param dictionary initiates a RF model
                        bestpcaoptionsdict = eval(
                            bestpcaoptions.ravel()[0])
                    # TODO POSSIBLE SOLUTION
                    #  add caveat if u get allof of missing because more dimensions than features
                    # because hypertuning is done on a larger dataset
                    pca = KernelPCA(**bestpcaoptionsdict)
                    Xpca = pd.DataFrame(data=pca.fit_transform(X_org))
                    Xpca.columns = ['pc' + str(x) for x in Xpca.columns]
                    Xpca.index = X_org.index
                    X = Xpca

                if paramsforclassifier is not False:
                    try:
                        print 'Unraveling params for classifier', paramsforclassifier
                        paramforclassifierdict = ast.literal_eval(
                            paramsforclassifier.ravel()[0])
                    except ValueError as e:
                        # This happens for model which initiates other models
                        #  such as Adaboost which param dictionary initiates a RF model
                        paramforclassifierdict = eval(
                            paramsforclassifier.ravel()[0])
                    # now set the classifier parameters from hyperparam tuning
                    classifier.set_params(**paramforclassifierdict)
                else:
                    pass

                # Diagnose purpose
                if paramsforclassifier is not False:
                    if get_params_from_regex:
                        print 'For', dataset, '\n and ', name, '\n using params for \n ', \
                            hyperparams.loc[
                                (hyperparams['dataset'].str.contains(hyperparamdfname)) &
                                (hyperparams['classifier'] == name), ['dataset', 'classifier']]
                    else:
                        print 'For', dataset, '\n and ', name, '\n using params for \n ', \
                            hyperparams.loc[
                                (hyperparams['dataset'] == dataset) &
                                (hyperparams['classifier'] == name), ['dataset', 'classifier']]
                else:
                    pass
                fitted = my_model_buildsaver(X, y, PATH, dataset,
                                             name, module_name, classifier)
    if in_production:
        print 'Running live predictions for production'
        predoutcomedict = {}
        for outcome in classifiers_by_label.keys():
            predoutcomedict['{}'.format(outcome)] = pd.DataFrame()
            
        for dataset, df in tqdm(dfd.iteritems(), total=len(dfd)):
            print '\n Processing {}'.format(dataset)
            # 2020-01-02_(LIVE|UPDATE)_treatment_label_week01_imputed
            # dataset = treatment_label_week01_imputed
            # path//..//models//2020-02-20_treatment_label_week01_imputed_LinearSVC.joblib
            timepoint = dataset.split('_')[2]
            if classifiers_by_label:
                outcome = [part for part in dataset.split('_')
                           if part in labeldictionary.keys()][0]
                classifiers = classifiers_by_label[outcome]

            # if you never have put individuals in here before
            # copy index from current df
            if len(predoutcomedict['{}'.format(outcome)]) == 0:
                predoutcomedict['{}'.format(outcome)] = pd.DataFrame(
                    index=df.index.copy())

            # make intermediate df containing df.index to append all the pred
            # to
            itermediatedf = pd.DataFrame(index=df.index.copy())

            if 'outcome' in df.columns:
                df = df.drop('outcome', axis='columns')

            for classifier in classifiers:
                # get name of classifier
                try:  # get name of quantile regression
                    if classifier.get_params()['loss'] == 'quantile':
                        name = '{}_{}'.format(
                            type(classifier).__name__,
                            ''.join(
                                str(classifier.get_params()['alpha']).split('.')
                            ))
                    else:
                        name = type(classifier).__name__
                except KeyError:
                    name = type(classifier).__name__


                # get newest created model of that name for this dataset
                classifierobjfile = newest(os.path.join(PATH, '..', 'models'),
                                   '.*{}.*{}_{}.joblib$'.format(module_name,
                                                         dataset,
                                                         name))
                # load it up
                classifierfit = load('{}'.format(classifierobjfile))

                # generate prediction
                # If there are patients in the dataset generate prediction
                if len(df) != 0:
                    if outcome == 'symptom-cont':
                        treatment = dataset.split('_')[0]
                        score_m, score_sd = meanstdforoutcome[treatment]
                        # get unscaled scores for
                        y_hat = (classifierfit.predict(df)*score_sd+score_m)
                    else:
                        y_hat = classifierfit.predict(df)
                    try:
                        y_hat_proba = classifierfit.predict_proba(df)[:, 1]
                    except Exception:
                        y_hat_proba = None
                else:  # if not patient
                    y_hat = np.nan
                    y_hat_proba = None

                # save the prediction in the dataframes itermediatedf
                # 'screen_LinearSVR'
                itermediatedf['{}_{}'.format(timepoint, name)] = y_hat
                try:
                    if isinstance(y_hat_proba, np.ndarray):
                        itermediatedf[
                            '{}_{}_{}'.format(timepoint, name, 'proba')] = (
                            y_hat_proba)
                    else:
                        pass
                except ValueError:
                    pass

            # merge all this timepoint/treatment
            if not predoutcomedict[outcome].index.name == itermediatedf.index.name:
                raise ValueError(predoutcomedict[outcome].index.name,
                                 itermediatedf.index.name,
                                 predoutcomedict[outcome],
                                 itermediatedf,
                                 'yhatproba is',
                                 y_hat_proba,
                                 'Merging dataframes does not have same index name')
            # combine first "patch values from object in the other -> if missing
            # in predoutcomedict[outcome] see if u have non NaN value in
            # itermediatedf then patch
            predoutcomedict[outcome] = predoutcomedict[outcome].combine_first(itermediatedf)
        print 'Saving prediction results and DST output to '
        print os.path.join(PATH, '..', 'results', 'production')
        for outcomename, predictionresults in predoutcomedict.iteritems():
            with open(os.path.join(PATH,'..','results','production',
                                   '{}_{}_{}.csv'.format(date,
                                                      module_name,
                                                      outcomename)),
                      'wb') as output:
                predictionresults.to_csv(path_or_buf=output,
                               encoding='utf8',
                               index=True,
                               mode='wb')
            preddftrim = predictionresults[[x for x in
                                            predictionresults.columns if
                                            re.search(r'.*GradientBoosting.*', x)]]

            if outcomename in labels_cont or outcomename == 'symptom-cont':
                preddftrim['modci'] = np.nan
                for times in weekly_split_dict.keys():
                    try:
                        preddftrim.loc[preddftrim['{}_GradientBoostingRegressor_025'.format(times)] >
                               preddftrim['{}_GradientBoostingRegressor'.format(times)],
                               'modci'] = (
                            np.where(
                                preddftrim.loc[preddftrim['{}_GradientBoostingRegressor_025'.format(times)] >
                               preddftrim['{}_GradientBoostingRegressor'.format(times)],
                               'modci'].isna(),
                                          times,
                                          preddftrim.loc[preddftrim['{}_GradientBoostingRegressor_025'.format(times)] >
                                                         preddftrim['{}_GradientBoostingRegressor'.format(times)],
                                                         'modci'].astype(str) +'_'+ times)
                        )
                    except KeyError as o:
                        print o
                        continue
                    # given you are above the 'mean' prediction with lower quanti.
                    # find these same again, then take your mean pred.
                    # redact how 'high above' the mean pred the lower quant was
                    try:
                        preddftrim.loc[
                        preddftrim['{}_GradientBoostingRegressor_025'.format(times)] >
                        preddftrim['{}_GradientBoostingRegressor'.format(times)],
                        '{}_GradientBoostingRegressor_025'.format(times)] = (
                            preddftrim.loc[preddftrim['{}_GradientBoostingRegressor_025'.format(times)] >
                                   preddftrim['{}_GradientBoostingRegressor'.format(times)],
                                           '{}_GradientBoostingRegressor'.format(times)] -
                            (preddftrim.loc[preddftrim['{}_GradientBoostingRegressor_025'.format(times)] >
                                   preddftrim['{}_GradientBoostingRegressor'.format(times)],
                                            '{}_GradientBoostingRegressor_025'.format(times)] - preddftrim.loc[
                                                                                                preddftrim['{}_GradientBoostingRegressor_025'.format(times)] >
                                                                                                preddftrim['{}_GradientBoostingRegressor'.format(times)],
                                                                                                '{}_GradientBoostingRegressor'.format(times)])
                        )
                    except KeyError as p:
                        continue
            else:
                pass

            with open(os.path.join(PATH, '..', 'results', 'production',
                                   '{}_{}_{}_{}.csv'.format(date,
                                                         module_name,
                                                         outcomename,
                                                            'plots')),
                      'wb') as output:
                preddftrim.to_csv(path_or_buf=output,
                                         encoding='utf8',
                                         index=True,
                                         mode='wb')





    print 'Done with prediction analyses'
    return None

##############################################################################
# Supporting functions
##############################################################################

def searchdfdict(substr,dictionary):
    """
    search dictionary for a key entry and return the value
    :param substr: what to look for
    :param dictionary: the dictionry to look in
    :return: the value corresponding to that key-hit
    """
    result = []
    for key in dictionary:
        if substr == key:
            result = dictionary[key]
    return result


def convertdatescols(df):
    """converting datecompleted_day into cos+sinetransform and
     converting datecompleted_time into the same (minutes used)"""
    print 'converting datecompleted_(day|time) into wave func*'
    dayscols = [x for x in df.columns if 'DateCompleted_day' in x]
    timecols = [x for x in df.columns if 'DateCompleted_time' in x]
    df[dayscols] = df[dayscols].astype('category')
    for col in dayscols:
        # get category codes then reinsert nan
        df[col] = df[col].cat.codes.replace(-1,np.nan)
        df[col] = (
                np.sin((df[col]) / 6 * (2 * np.pi))
                + np.cos((df[col]) / 6 * (2 * np.pi))
        )
    for col in timecols:
        try:
            df[col] = (
                    np.sin((df[col].dt.hour*60+df[col].dt.minute)/1440*(2*np.pi))
                    + np.cos((df[col].dt.hour*60+df[col].dt.minute)/1440*(2*np.pi))
            )
        except AttributeError:
            df[col] = pd.to_datetime(df[col].astype(str))
            df[col] = (
                    np.sin((df[col].dt.hour * 60 + df[col].dt.minute) / 1440 * (2 * np.pi))
                    + np.cos((df[col].dt.hour * 60 + df[col].dt.minute) / 1440 * (2 * np.pi))
            )
    return df


def mainsymptommerge(row,i):
    """Taking in a row and checking which treatment you belong to and
        returns a scaled version of primary symptom accordingly """
    if row['Treat_org_drop'] == 'Panic':  # and i in ['SCREEN','PRE']: #add conditional b.c. aggregating week easier
        return row['PDSS-SR-3064_{}_sum'.format(i)]
    if row['Treat_org_drop'] == 'Depression':
        return row['MADRS-1951_{}_sum'.format(i)]
    if row['Treat_org_drop'] == 'Social_Anxiety':  # and i in ['SCREEN','PRE']:
        return row['LSAS-2241_{}_sum'.format(i)]

def paramdict(x):
    x = re.sub(r'(\n+)(?= )| ', '', x)
    try:
        x = ast.literal_eval(x)
    except ValueError as e:
        pass
    return ast.literal_eval(x)

def newest(path,pattern):
    """
    Get newest file by the regex pattern in designated path 1 file
    :param path:
    :param pattern:
    :return:
    """
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files if
             re.search(pattern,basename)]
    return max(paths, key=os.path.getctime)


if __name__ == "__main__":
   pass