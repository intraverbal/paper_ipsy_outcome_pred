#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module take in information from arbetsbladen, insstruments and outcome
merges them together, handles missing and standardisation of variables (which
 may or may not be saved and finally hypertunes (optional) and runs analyses
Relies heavily on other modules - especially prediction (for analyses)
hyperparamtuning (for hypertuning) and prediction for some data cleaning and
aux functions
"""

import numpy as np
import re
import pandas as pd
import datetime
import os
import warnings
import csv
import ast
import sys

from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from missingpy import MissForest
from joblib import dump, load


from prediction import *
from modules.cleandates import cleandates
from modules.hyperparamtuning import *
from modules.readfile import *
from modules.datacleaner import mainmerger
from modules.datacleaner import convertdatescols
from modules.datacleaner import exr_clean_weekX
from modules.datacleaner import reloaddfs, reloaddf, savedfs, newest


CONTOUTCOME = True
PATH = ''
date = datetime.datetime.today()
#VERBOSE = True
#MISSMIN = True

##############################################################################
# Mering functions (outcome,instruments and arbetsblad)
##############################################################################


def mergedfssetupfeatures_pca(outcome, arbetsblad,
                              instruments, whatoutcome=outcomedict):
    """
    1. Minimzing the features and creating main symptoms. Modified by pca
    by outsouring some jobs later and minimzing symptoms in other modules
    :param outcome: outcome dataframe
    :param arbetsblad: arbetsblad_wide_interpreted
    :param instruments: instruments_filtered_merged_scored_labeled
    :param whatoutcome:outcome dictionary for how to interpret dichotomization
    :return: a df with above dataframes fused into one + dflabeloutcome
    dataframe containing all information needed to label a patient from a
    continous prediction (patient info, the pre-treatment
    main symptom score, the treatment patient was in, the dicotomized score)
    """
    print 'Merging outcome,arbetsblad and instruments pca'
    todrop = []
    # fix some col. names #these neeed to exist in the namespace
    df = cleanoutcome_pca(outcome)  # get out everything we need and dropsome

    # need to drop these columns for later
    todrop.extend([x for x in df.columns if re.search(
        r'(1951|2241|3064)_POST_sum|'
        r'^.*_label([0-9]|cont)', x)])

    # Rename treatments first
    df['Treatment'] = df['Treatment'].replace(
        {'Depression STARTA EJ': 'Depression',
         'Depression 2.0': 'Depression',
         'Paniksyndrom': 'Panic',
         'Social fobi STARTA EJ': 'Social_Anxiety',
         'Social fobi 2.0': 'Social_Anxiety'})

    # fixing upp sex
    df['sex'] = df['sex'].map({'M': 1, 'F': 0})

    # Make labels and treatment indicatiors

    # Treatment indicator
    #
    # if u wanna keep 'Treatment' org col 4 later
    df.loc[:, 'Treat_org_drop'] = df.loc[:, 'Treatment']
    df = pd.get_dummies(df,
                   columns=['Treatment'],
                   drop_first=False, prefix='Treatment')
    df.columns = [col.replace('Treatment_', '') for col in df.columns]

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
        # save dictionary outcome for later scoring properties deprecated 
        df.loc[df['Treat_org_drop'] == 'Panic', 'outcomesave'] = (
            df.loc[df['Treat_org_drop'] == 'Panic', 'PDSS-SR-3064_label7'])
        df.loc[df['Treat_org_drop'] == 'Depression', 'outcomesave'] = (
            df.loc[df['Treat_org_drop'] == 'Depression', 'MADRS-1951_label10'])
        df.loc[df['Treat_org_drop'] == 'Social_Anxiety', 'outcomesave'] = (
            df.loc[df['Treat_org_drop'] == 'Social_Anxiety', 'LSAS-2241_label34'])
    else:
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

    # create outcome
    if CONTOUTCOME:
        print 'Using', 'continous symptom', 'as dictoutcome'
        print ('NOT removing those with unknown outcome n=',
               df['outcome'].isnull().sum())
        #df = df.loc[~df['outcome'].isnull(), :]
        df['outcomesave'] = df['outcomesave'].replace(outcomedict)

        # Generating a seperate dataframe to keep track of both continous
        # and the dichotomized outcome for later labling
        colforlabling = ['Patient', 'Treat_org_drop', 'outcome', 'outcomesave']
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
        exit('not doing dict outcome')
        print 'Using', whatoutcome, 'as dictoutcome'
        df['outcome'] = df['outcome'].replace(whatoutcome)
        print ('removing those with unknown outcome n=',
            len(df[df['outcome'] == 'unknown']))
        df = df.loc[df['outcome'] != 'unknown', :]

    # Gathering info from other sources (arbetsblad and instruments)

    # Arbetsblad

    pickedarbetsblad = cleanarbetsbladen(arbetsblad)
    df = df.merge(pickedarbetsblad, how='left', on=u'Patient')

    # instruments
    pickedinstruments, namesfrominstruments = cleaninstruments_pca(instruments)
    df = df.merge(pickedinstruments, how='left', on = u'Patient')

    # Set index before all the splitting begins
    df.set_index('Patient', inplace=True)

    print 'Done!'
    if dflabeloutcome is None:
        dflabeloutcome = pd.DataFrame()
    writefile_pca(dflabeloutcome,'dflabeloutcome')
    return df.drop(todrop, axis='columns'), dflabeloutcome

def merged_cleansymptoms_pca(df):
    """
    takes in a merged dataframe (from mergedfssetupfeatures_pca) and cleans
    symptoms, standardisez variables and creates dummy variables if necessaryy
    PCA different because allof of other variables.
    :param df:
    :return: dataframe and a dictionary for transforming back outcome variable.
    """
    print 'Extracting, cleaning and standardizing variables'
    todrop = []

    # Begin by standardising columns
    df = pd.get_dummies(df, dummy_na = True,
                        columns=[u'Anamnes-(ur-SCID)-1827_SCREEN_1833_2a',
                                 u'Anamnes-(ur-SCID)-1827_SCREEN_1843_5',
                                 u'Gticp-0_SCREEN_0_adr',
                                 u'Gticp-0_SCREEN_0_aallpsychs',
                                 u'Gticp-0_SCREEN_0_bassist',
                                 u'Gticp-0_SCREEN_0_healer',
                                 u'GSticp-0_SCREEN_0_bgoogle',
                                 u'Gticp-0_SCREEN_0_bfamily',
                                 u'Gticp-0_SCREEN_0_difworkmerge'],
                        drop_first=True, prefix=['Marital_1833',
                                                 'Edu_1843',
                                                 'Gticp_adr',
                                                 'GSticp_apsych',
                                                 'Gticp_bassist',
                                                 'Gticp_healer',
                                                 'GSticp_bgoogle',
                                                 'Gticp_bfamiliy',
                                                 'Gticp_work'])

    # Special columns that need attention
    df[u'GNybesök-0_PRE_0_clinra-GAF'] = (
        df[u'GNybesök-0_PRE_0_clinra-GAF'].str.replace(r'[^\w\s]', ''))


    # don't standardise these
    listnotstand = [x for x
                   in df.columns
                   if re.search(r'(DateCompleted|POST|sex|^Geq5d.*(?<!_duration)$|'
                                r'^Geq5d-.*(?<!_sum)$|^GSeq5d.*(?<!_sum)$|'
                                r'GSeq5d-0(?<!_VAS)$|Gticp*(?<!_duration)|'
                                r'GSticp*(?<!_duration)|'
                                r'^GNybes.*-0.*(?<!_clinra-GAF)|label[0-9]|'
                                r'^GSNybes.*-0.*(?<!_clinra-GAF)|1827|'
                                r'Treat_org_drop|Depression|Panic|Social_Anxiety|'
                                r'outcome|^TreatmentAccess(Start|End)$|^HW-|'
                                r'Edu_1843|Marital_1833)',
                                x)]
    listnotstand = [x for x in listnotstand if not
    re.search(r'^(G|GS)eq5d.*(_duration|sum|VAS)$',x)]
    todrop.extend([x for x in df.columns if
                   re.search(r'(G|GS)Nybes.*-0.*_(duration|DateCompleted(|_time|_day))$',x)])
    listtostand = list(set(df.columns) - set(listnotstand))
    listtostanduse = [x for x in listtostand]

    for col in listtostand:
        # make string -> # precaution against missing ->  missings are false
        # -> if there are no strings ignore
        try:
            if df.loc[:, col].\
                astype(unicode, skipna=True).\
                replace('nan', np.nan).\
                str.contains('[A-za-z]', na=False).\
                    eq(False).all():
                pass
            else:
                listtostanduse.remove(col)  # don't standardise it either
                todrop.append(col)  # if column contain strings skip it
        except AttributeError:
            print 'For col', col, 'are everyone missing?'
            print 'miss is: ', df[col].isna().sum()
            print 'removing column'
            listtostanduse.remove(col)
            todrop.append(col)

    # not standardised but keep these variables:
    notstandkeep = ['Treat_org_drop','outcome',
                    'TreatmentAccessStart', 'TreatmentAccessEnd']
    # also not standardised datecompleted or label because they have special
    # formatting
    notstandkeep.extend([x for x in listnotstand if 'DateCompleted' in x or 'label' in x])
    # Find those columns which we dont wanna standardise but also don't want
    # to keep - then search these for columns containing strings and rm them
    purgenotstand = list(set(listnotstand) - set(notstandkeep))
    notstandkeepnotdates = list(set(notstandkeep) -
                                set(
                                    [x for x in notstandkeep if
                                     re.search(r'DateCompleted',x)]))
    testpurge = []
    for col in purgenotstand:
        # make string -> # precaution against missing ->  missings are false
        # -> if there are no strings ignore
        try:
            if df.loc[:, col]. \
                    astype(unicode, skipna=True). \
                    replace('nan', np.nan). \
                    str.contains('[A-za-z]', na=False). \
                    eq(False).all():
                pass
            else:
                #testpurge.append(col)
                todrop.append(col)  # if column contain strings skip it
        except AttributeError:
            print 'For col',col,'are everyone missing?'
            print 'miss is: ', df[col].isna().sum()
            print 'removing column'
            todrop.append(col)

    postvariables = [x for x in df.columns if 'POST' in x]
    df_postvars = df.loc[:,postvariables]
    fuvars = [x for x in df.columns if '_FU' in x]
    df_fuvars = df.loc[:, fuvars]
    todrop.extend(postvariables)
    todrop.extend(df_fuvars)
    listtostanduse = [x for x in listtostanduse
                      if x not in postvariables and x not in fuvars]
    # adding some drops
    todrop.extend(['TreatmentAccessStart', 'TreatmentAccessEnd'])

    # dropping then continuing
    df.drop(todrop,inplace=True,axis='columns')

    # By these standardising now everything is either z-transformed or
    # scaled as according to dates - OR between 0-1 b.c. of nature of quest.
    df[listtostanduse] = StandardScaler().fit_transform(df[listtostanduse])

    # converting dates and standardising them
    df = convertdatescols(df)

    # DatesCompleted standardised
    # standardise all DateCompleted (not time and day)
    allDateCompleted = [x for x in df.columns if
                        re.search(r'.*_DateCompleted$',x)]
    df[allDateCompleted] = StandardScaler().fit_transform(df[allDateCompleted])

    #

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
                                       'datasets', 'processed', 'study1','PCA')):
            print 'Result folder does not exist, creating results folder'
            os.makedirs(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1','PCA'))
        print 'Saving means and std for later for', meanstdforoutcome.keys()
        for name, values in meanstdforoutcome.iteritems():
            with open(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1','PCA',
                      '{}_{}_{}.txt'.format(date.strftime('%Y-%m-%d'),
                                        name,
                                        'meanstdforoutcome')), 'wb') as output:
                textout = ','.join([str(x) for x in values])
                output.write(textout.encode('utf8'))

    #  do final numeric representation of all variables
    for col in df.columns:
        if col == 'Treat_org_drop': #gotta skip this one or later breaks
            continue
        elif df[col].dtype == 'object':
            try:
                df.loc[:,col] = pd.to_numeric(df.loc[:,col], errors='coerce')
            except Exception as e:
                print 'error in trying to make numeric',col
        else:
            pass
    print 'Done!'
    return df, meanstdforoutcome


def splitintoweekwise_pca(df):
    """ split the dataset into weekly dataset return a dictionary
    of datasets heavily reliant on names of columns and exr_clean_weekX
    different than prediction.py because it has other columns
    :param df: a dataframe
    :return: a dictionary of dataframes split by week (screen - week04/day 28)
    """
    dfdict = {}
    # -xx days
    names_screen = [x for x
                   in df.columns
                   if re.search(r'(outcome|1843|1833|^(G|GS)ticp_|'
                                r'currentwork|ncomorbid|'
                                r'^Treat.*|Depression|Panic|Social_Anxiety|'
                                r'sex|age$|'
                                r'SCREEN)', x)]

    dfdict['All_screen'] = df.loc[:,names_screen]
    # around day 0
    names_pre = [x for x
                   in df.columns
                   if re.search(r'(outcome|1843|1833|^(G|GS)ticp_|'
                                r'currentwork|ncomorbid|'
                                r'^Treat.*|Depression|Panic|Social_Anxiety|'
                                r'sex|age$|'
                                r'SCREEN|PRE)', x)]
    dfdict['All_pre'] = df.loc[:, names_pre]
    # around day 7
    # Due note that we do the first 7 days of treatment as week 1 (not 0)
    names_week01 = [x for x
                in df.columns
                if re.search(r'(outcome|1843|1833|^(G|GS)ticp_|'
                             r'currentwork|ncomorbid|'
                             r'^Treat.*|Depression|Panic|Social_Anxiety|'
                             r'sex|age$|'
                             r'SCREEN|PRE|^(messages|homeworks).*_7$)', x)]
    dfdict['All_week01'] = exr_clean_weekX(df, names_week01, 6)
    # around day 14
    names_week02 = [x for x
                    in df.columns
                    if re.search(r'(outcome|1843|1833|^(G|GS)ticp_|'
                                 r'currentwork|ncomorbid|'
                                 r'^Treat.*|Depression|Panic|Social_Anxiety|'
                                 r'sex|age$|'
                                 r'SCREEN|PRE|MID|.*WEEK01.*|^HW-01.*|'
                                 r'^(messages|homeworks).*_(7|14)$)', x)]
    # exr_clean_weekX restricts info by week if it happens to have been found
    # to been tampered with in hindsight
    dfdict['All_week02'] = exr_clean_weekX(df, names_week02, 13)
    # around day 21
    names_week03 = [x for x
                    in df.columns
                    if re.search(r'(outcome|1843|1833|^(G|GS)ticp_|'
                                 r'currentwork|ncomorbid|'
                                 r'^Treat.*|Depression|Panic|Social_Anxiety|'
                                 r'sex|age$|'
                                 r'SCREEN|PRE|MID|.*WEEK(01|02).*|^HW-0[1-2].*|'
                                 r'cscale|'
                                 r'^(messages|homeworks).*_(7|14|21)$)', x)]
    dfdict['All_week03'] = exr_clean_weekX(df, names_week03, 20)
    # Day 28 last
    names_week04 = [x for x
                    in df.columns
                    if re.search(r'(outcome|1843|1833|^(G|GS)ticp_|'
                                 r'currentwork|ncomorbid|'
                                 r'^Treat.*|Depression|Panic|Social_Anxiety|'
                                 r'sex|age$|'
                                 r'SCREEN|PRE|MID|.*WEEK(01|02|03).*|^HW-0[1-3].*|'
                                 r'cscale|'
                                 r'^(messages|homeworks).*_(7|14|21|28)$)', x)]
    dfdict['All_week04'] = exr_clean_weekX(df, names_week04, 27)
    return dfdict


def splitintotreatments_pca(dfdict, missing=False):
    """
    splits a dataframe dictionary into a larger dictionary also containing
    splits for each treatment. Different for pca b.c. also dropping the
    global dataset because we cannot use merged symptom in the same way.
    It also purges all columns with more than 90% missing cases
    :param dfdict: dictionary of dataframes (weekise splitted)
    :param missing: is there any missing in the dataset (not used)
    :return: a bigger dictionary including dataframes spliited by treatment
    """
    print 'splitting into treatment specifics keeping features ' \
          'dropping'
    dfdictout = {}
    todrop = ['Treat_org_drop','Depression','Panic','Social_Anxiety',
              'HW-01_ChangedDayFromStart', 'HW-02_ChangedDayFromStart',
               'HW-03_ChangedDayFromStart']
    print 'drop these', todrop
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

def missingwrapper_pca(dfdict):
    if not os.path.exists(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1',
                                       'PCA', 'Alldata')):
        print 'Result folder does not exist, creating results folder'
        os.makedirs(os.path.join(PATH, '..',
                                 'datasets', 'processed', 'study1', 'PCA',
                                 'Alldata'))
    savedfs(dfdict, os.path.join(PATH, '..',
                                                'datasets', 'processed',
                                                'study1', 'PCA',
                                                'Alldata'),
            date=date.strftime('%Y-%m-%d'))

    print 'creating split for test and train'
    smallerdict = {}
    for name, df in dfdict.iteritems():
        smallerdict['{}'.format(name)] = df.sample(frac=0.1,
                                                   random_state=0)
        dfdict[name] = (
            df.drop(smallerdict[name].index)
        )
    print 'Saving all dataframes in ', os.path.join(PATH, '..',
                                                    'datasets',
                                                    'processed', 'study1')
    if not os.path.exists(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1',
                                       'PCA', 'train')):
        print 'Result folder does not exist, creating results folder'
        os.makedirs(os.path.join(PATH, '..',
                                 'datasets', 'processed', 'study1', 'PCA',
                                 'train'))
    savedfs(dfdict, os.path.join(PATH, '..',
                                  'datasets', 'processed', 'study1', 'PCA',
                                  'train'),
            date=date.strftime('%Y-%m-%d'))
    if not os.path.exists(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1',
                                       'PCA', 'test')):
        print 'Result folder does not exist, creating results folder'
        os.makedirs(os.path.join(PATH, '..',
                                 'datasets', 'processed', 'study1', 'PCA',
                                 'test'))
    savedfs(smallerdict, os.path.join(PATH, '..',
                                      'datasets', 'processed', 'study1',
                                      'PCA', 'test'),
            date=date.strftime('%Y-%m-%d'))

    dfdictmissfixtrain = reloaddfs(regexfile=r'.*_.*_.*.csv',
                            fullPATH=os.path.join(PATH, '..',
                                                  'datasets', 'processed',
                                                  'study1',
                                                  'PCA',
                                                  'train'))
    dfdictmissfixfixedtrain = missingremoval_pca(dfdictmissfixtrain,
                                            dfdicttrain={},
                                            training=True)
    if not os.path.exists(os.path.join(PATH, '..',
                                       'datasets',
                                       'processed',
                                       'study1',
                                       'PCA',
                                       'train',
                                       'missfix')):
        os.makedirs(os.path.join(PATH, '..',
                                 'datasets', 'processed',
                                 'study1',
                                 'PCA',
                                 'train',
                                 'missfix'))
    savedfs(dfdictmissfixfixedtrain, os.path.join(PATH,
                                             '..',
                                             'datasets',
                                             'processed',
                                             'study1',
                                             'PCA',
                                             'train',
                                             'missfix'),
            date=date.strftime('%Y-%m-%d'),
            in_production=False)
    # TEST
    dfdictmissfixtest = reloaddfs(regexfile=r'.*_.*_.*.csv',
                              fullPATH=os.path.join(PATH, '..',
                                                    'datasets', 'processed',
                                                    'study1',
                                                    'PCA',
                                                    'test'))
    dfdictmissfixfixedtest = missingremoval_pca(dfdictmissfixtest,
                                                 dfdicttrain=dfdictmissfixfixedtrain,
                                                 training=False)
    if not os.path.exists(os.path.join(PATH, '..',
                                       'datasets',
                                       'processed',
                                       'study1',
                                       'PCA',
                                       'test',
                                       'missfix')):
        os.makedirs(os.path.join(PATH, '..',
                                 'datasets', 'processed',
                                 'study1',
                                 'PCA',
                                 'test',
                                 'missfix'))
    savedfs(dfdictmissfixfixedtest, os.path.join(PATH,
                                                  '..',
                                                  'datasets',
                                                  'processed',
                                                  'study1',
                                                  'PCA',
                                                  'test',
                                                  'missfix'),
            date=date.strftime('%Y-%m-%d'),
            in_production=False)
    return dfdictmissfixfixedtrain

def missingremoval_pca(dfdict,dfdicttrain,training=True):
    """
    removes all missing in the dataframe according to a mix of manual
     and auotmatic rules . Different for pca because we need to handle other
     columns. Uses a fixed number of 25% missing maximum to keep column.
    :param dfdict: a dictionary of dataframes
    :return: dictionary of dataframes with no missing - 2 different methods
    one containing imputed values and one removed cases. (Both removed columns
    with 25% missing or more). Also WRITES these dataframes as .csv to designated
    path + subfolders
    """
    print 'Handling missing'
    dictout = dict()
    for name, df in dfdict.iteritems():
        # Manual filling
        if name.startswith('All'):
            continue
        df = df.loc[~df['sex'].isna(), :]

        colstofill = [x for x in df.columns if x in
                      ['HW-01', 'HW-02', 'HW-03',
                       'ncomorbid']]

        #df[colstofill] = df[colstofill].fillna(-99)

        #print 'Imputing missing using rf'
        if training:
            # Mass removal
            maskkeep = (df.loc[:, :].isnull().mean() < 0.25)
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
                                               'PCA',
                                               'missimputeobj')):
                print 'Missimputeobj folder does not exist, creating folder'
                os.makedirs(os.path.join(PATH, '..',
                                         'datasets', 'processed',
                                         'study1',
                                         'PCA',
                                         'missimputeobj'))
            try:
                dump(imputer,
                     os.path.join(PATH, '..',
                                  'datasets', 'processed',
                                  'study1',
                                  'PCA',
                                  'missimputeobj',
                                  '{}_{}_{}.joblib'.format(
                                      date.strftime('%Y-%m-%d'),
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
                    namingna = '{}-{}_{}'.format(
                        '_'.join(name.split('_')[0:2]),
                        'naremove',
                        '_'.join(name.split('_')[2:]))
                else:
                    namingna = '{}-naremove'.format(name)
                maskkeepna = dfdicttrain[namingna].columns
            except KeyError as ek:
                print 'keys are', dfdicttrain.keys()
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

            try:
                if 'benchmark' in name.split('_'):
                    naming1 = '{}-{}_{}'.format('_'.join(name.split('_')[0:2]),
                                                'imputed',
                                                '_'.join(name.split('_')[2:]))
                else:
                    naming1 = '{}-imputed'.format(name)
                maskkeep = dfdicttrain[naming1].columns
            except KeyError as ek:
                print 'keys are', dfdicttrain.keys()
                print ek, 'occured'
                quit()
            df = df.loc[:, maskkeep]

            imputerfile = newest(os.path.join(PATH, '..',
                                              'datasets', 'processed',
                                              'study1',
                                              'PCA',
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
        dictout['{}'.format(name+'-imputed')] = df
        dictout['{}'.format(name+'-naremove')] = dfalmostnoimpute
    return dictout


##############################################################################
# Supporting functions
##############################################################################



def isetdiff(n1,n0):
    """
    auxilary function to see difference in names across datasets.
    :param n1:
    :param n0:
    :return:
    """
    l = list(set(n1)-set(n0))
    print '"diff are:"'
    for x in l:
        print x
    print 'l of diff',len(l)

def writefile_pca(df,name):
    print 'Saving all dataframe in ', os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1',
                                       'PCA')
    if not os.path.exists(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1',
                                       'PCA')):
        print 'Result folder does not exist, creating results folder'
        os.makedirs(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1',
                                       'PCA'))
    with open(os.path.join(PATH, '..',
                                       'datasets', 'processed', 'study1',
                                       'PCA',
              '{}_{}.csv'.format(date.strftime('%Y-%m-%d'),
                                           name)), 'wb') as output:
        df.to_csv(path_or_buf=output,
                  encoding='utf8',
                  index=True,
                  mode='wb')



if __name__ == "__main__":
    pass