'''This module provides Python/statsmodel solutions to all code examples in

Dobson AJ & Barnett AG: "An Introduction to Generalized Linear Models"
3rd ed
CRC Press(2008)

Points that still need to be done are marked with "tbd" below:

- [Error 1] cloglog values in "logistic_regression" are wrong 

- [Unclear] in "senility_and_WAIS" I don't understand what the "grouped response"
  is supposed to mean

- [Error 2] the signs of the paramters in "nominal_logistic_regression" are
  incorrect

- [Missing 1] "ordinal_logistic_regression" is not yet implemented in statsmodels

- [Error 3] the standard errors in "poisson_regression" are wrong

- [Missing 2] Cox proportional hazards are not yet implemented in statsmodels

- [Missing 3] Repeated measures models are not yet implemented in statsmodels

author: thomas haslwanter
date:   may 2013
ver:    0.2

'''

# Standard libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

from statsmodels.stats.api import anova_lm

# for data import
import urllib2
import zipfile
from StringIO import StringIO

def get_data(inFile):
    '''Get data from original Excel-file'''
    
    # get the zip-archive
    url = 'http://cdn.crcpress.com/downloads/C9500/GLM_data.zip'
    GLM_archive = urllib2.urlopen(url).read()
    
    # extract the requested file from the archive, as a pandas XLS-file
    
    zipdata = StringIO()
    zipdata.write(GLM_archive)
    myzipfile = zipfile.ZipFile(zipdata)
    xlsfile = myzipfile.open(inFile)
    xls = pd.ExcelFile(xlsfile)
    df = xls.parse('Sheet1', skiprows=2)    
    
    return df

def regression():
    '''Poisson regression example
    chapter 4.4, p.69'''
    
    # get the data from the web
    inFile = r'GLM_data/Table 4.3 Poisson regression.xls'
    df = get_data(inFile)
    
    # do the fit
    p = sm.GLM.from_formula('y~x', family=sm.families.Poisson(sm.families.links.identity), data=df)
    print p.fit().summary()    

def multiple_linear_regression():
    '''Multiple linear regression
    chapter 6.3, p. 98'''
    
    # get the data from the web
    inFile = r'GLM_data/Table 6.3 Carbohydrate diet.xls'
    df = get_data(inFile)
    
    # do the fit, for the original model ...
    model = sm.OLS.from_formula('carbohydrate ~ age + weight + protein', data=df).fit()
    print model.summary()
    print anova_lm(model)

    # as GLM
    glm = sm.GLM.from_formula('carbohydrate ~ age + weight + protein',
            family=sm.families.Gaussian(), data=df).fit()
    print 'Same model, calculated with GLM'
    ''' The confidence intervals are different than those from OLS.
    The reason (from Nathaniel Smith):
    OLS uses a method that gives exact results, but only works in the special
    case where all the usual OLS criteria apply - iid Gaussian noise etc. GLM
    instead uses an approximate method which is correct asymptotically but may
    be off for small samples; the tradeoff you get in return is that this method
    works the same way for all GLM models, including those with non-Gaussian
    error terms and non-trivial link functions. So that's why they're different.
    '''

    print glm.summary()
    
    # ... and for model 1
    model1 = sm.OLS.from_formula('carbohydrate ~ weight + protein', data=df).fit()
    print model1.summary()
    print anova_lm(model1)    

def anova():
    '''ANOVA
    chapter 6.4, p. 108, and p. 113
    GLM does not work with anova_lm.
    '''
    
    # get the data from the web
    inFile = r'GLM_data/Table 6.6 Plant experiment.xls'
    df = get_data(inFile)
    
    # fit the model (p 109)
    glm = sm.GLM.from_formula('weight~group', family=sm.families.Gaussian(), data=df)
    print glm.fit().summary()        
    
    print '-'*65
    print 'OLS'
    model = sm.OLS.from_formula('weight~group', data=df)
    print model.fit().summary()
    print anova_lm(model.fit())            
    
    # The model corresponding to the null hypothesis of no treatment effect is
    model0 = sm.OLS.from_formula('weight~1', data=df)
    
    # Get the data for the two-factor ANOVA (p 113)
    inFile = r'GLM_data/Table 6.9 Two-factor data.xls' 
    df = get_data(inFile)
    
    # adjust the header names from the Excel-file
    df.columns = ['A','B', 'data']
    
    # two-factor anova, with interactions
    ols_int = sm.OLS.from_formula('data~A*B', data=df)
    anova_lm(ols_int.fit())
    
    # The python commands for the other four models are
    ols_add = sm.OLS.from_formula('data~A+B', data=df)
    ols_A = sm.OLS.from_formula('data~A', data=df)    
    ols_B = sm.OLS.from_formula('data~B', data=df)    
    ols_mean = sm.OLS.from_formula('data~1', data=df)    

def ancova():
    ''' ANCOVA
    chapter 6.5, p 117 '''
    
    # get the data from the web
    inFile = r'GLM_data/Table 6.12 Achievement scores.xls'
    df = get_data(inFile)
    
    # fit the model
    model = sm.OLS.from_formula('y~x+method', data=df).fit()
    print anova_lm(model)
    print model.summary()    

def logistic_regression():
    '''Logistic regression example
    chapter 7.3, p 130
    [tbd]: cloglog values are incorrect
    '''
    
    inFile = r'GLM_data/Table 7.2 Beetle mortality.xls'
    df = get_data(inFile)
    
    # adjust the unusual column names in the Excel file
    colNames = [name.split(',')[1].lstrip() for name in df.columns.values]
    df.columns = colNames
    
    # fit the model
    df['tested'] = df['n']
    df['killed'] = df['y']
    df['survived'] = df['tested'] - df['killed']
    model = sm.GLM.from_formula('survived + killed ~ x', data=df, family=sm.families.Binomial()).fit()
    print model.summary()
    
    print '-'*65
    print 'Equivalent solution:'
    
    model = sm.GLM.from_formula('I(n - y) + y ~ x', data=df, family=sm.families.Binomial()).fit()
    print model.summary()    
    
    # The fitted number of survivors can be obtained by
    fits = df['n']*(1-model.fittedvalues)
    print 'Fits Logit:'
    print fits
    
    # The fits for other link functions are:
    model_probit = sm.GLM.from_formula('I(n - y) + y ~ x', data=df, family=sm.families.Binomial(sm.families.links.probit)).fit()
    print model_probit.summary()
    
    fits_probit = df['n']*(1-model_probit.fittedvalues)
    print 'Fits Probit:'
    print fits_probit
    
    model_cll = sm.GLM.from_formula('I(n - y) + y ~ x', data=df, family=sm.families.Binomial(sm.families.links.cloglog)).fit()
    print model_cll.summary()
    fits_cll = df['n']*(1-model_cll.fittedvalues)
    print 'Fits Extreme Value:'
    print fits_cll

def general_logistic_regression():
    '''Example General Logistic Recression,
    Example 7.4.1, p. 135'''
    
    # Get the data
    inFile = r'GLM_data/Table 7.5 Embryogenic anthers.xls'
    df = get_data(inFile)
    
    # Define the variables so that they match Dobson
    df['n_y'] = df['n'] - df['y']
    df['newstor'] = df['storage']-1
    df['x'] = np.log(df['centrifuge'])
    
    # Model 1
    model1 = sm.GLM.from_formula('n_y + y ~ newstor*x', data=df, family=sm.families.Binomial()).fit()
    print model1.summary()
    
    # Model 2
    model2 = sm.GLM.from_formula('n_y + y ~ newstor+x', data=df, family=sm.families.Binomial()).fit()
    print model2.summary()
    
    # Model 3
    model3 = sm.GLM.from_formula('n_y + y ~ x', data=df, family=sm.families.Binomial()).fit()
    print model3 .summary()    

def senility_and_WAIS():
    '''Another example of logistic regression.
    chapter 7.8, p 143
    [tbd]: I don't understand how the "Binomial model" (grouped response)
    is supposed to work, in either language'''

    inFile = r'GLM_data/Table 7.8 Senility and WAIS.xls'
    df = get_data(inFile)
    
    # ungrouped
    model = sm.GLM.from_formula('s ~ x', data=df, family=sm.families.Binomial()).fit()
    print model.summary()    
    
    # Hosmer-Lemeshow
    # grouped: Here I don't get how the grouping is supposed to be achieved, either in R or in Python
    # [tbd]

def nominal_logistic_regression():
    '''Nominal Logistic Regression
    chapter 8.3,  p. 155 
    
    At this point, I nominal logistic regression can be done with the formula approach.
    
    Regarding the output, note that R produces log(pi2/pi1) and log(pi3/pi1), while
    statsmodels produces log(pi2/pi1) and log(pi3/pi2) 
    '''
    
    # Get the data
    inFile = r'GLM_data/Table 8.1 Car preferences.xls'
    df = get_data(inFile)    
    
    # Generate the design matrices using patsy
    pm = patsy.dmatrices('response~age+sex', data=df)
    
    # Change the first output, representing the endogenous data, into a vector
    # e.g. [0,1,0] -> 1
    # e.g. [0,0,1] -> 2
    endog_ind = np.zeros(len(pm[0]))
    for ii in range(len(pm[0])):
        endog_ind[ii] = np.where(pm[0][ii])[0]

    # Since frequencies cannot be represented explicitly, multiply the entries
    # to correspond to the correct number of occurences
    endog = np.repeat(endog_ind, df['frequency'].values.astype(int), axis=0)
    # equivalent to
    # endog = np.repeat(df['response'], df['frequency'].values.astype(int), axis=0) 

    exog = np.array(np.repeat(pm[1], df['frequency'].values.astype(int), axis=0))
    
    # Fit the model, and print the summary
    model = sm.MNLogit(endog, exog, method='nm').fit()
    print  model.summary()

    #[tbd] some of the model parameters have the wrong sign - all signs
    # are the same
    
def ordinal_logistic_regression_tbd():
    
    '''Ordinal Logistic Regression
    chapter  8.4, p161
    This function is not implemented in statsmodels yet. One solution can be found at
    http://fabianp.net/blog/2013/logistic-ordinal-regression/
    '''
    
    inFile = r'GLM_data/Table 8.1 Car preferences.xls'
    df = get_data(inFile)    

def poisson_regression():
    '''Poisson Regression
    chapter 9.2, p.170 & 171 '''
    
    inFile = r"GLM_data/Table 9.1 British doctors' smoking and coronary death.xls"
    df = get_data(inFile)    
    print df

    # Generate the required variables
    df['smoke'] = np.zeros(len(df))
    df['smoke'][df['smoking']=='smoker']=1

    df['agecat'] = np.array([1,2,3,4,5,1,2,3,4,5])
    df['agesq'] = df['agecat']**2

    df['smkage'] = df['agecat']
    df['smkage'][df['smoking']=='non-smoker']=0

    df['death_py'] = df['deaths']/df['person-years']

    model = sm.GLM.from_formula('death_py~agecat+agesq+smoke+smkage', family=sm.families.Poisson(), data=df).fit()
    print model.summary()

    #[tbd]: standard errors are wrong


def log_linear_models():
    '''Log-linear models
    chapter 9.7, p 180 & 182 '''

    # Malignant melanoma, p 180 --------------------------------
    inFile = r'GLM_data/Table 9.4 Malignant melanoma.xls'
    df = get_data(inFile)    

    # Minimal model
    model_min = sm.GLM.from_formula('frequency~1', family = sm.families.Poisson(), data=df).fit()
    print 'Malignant melanoma'
    print model_min.fittedvalues[0]

    # Additive model
    model_add = sm.GLM.from_formula('frequency~site+type', family = sm.families.Poisson(), data=df).fit()
    print model_add.fittedvalues[0]

    # Saturated model
    # model_sat = sm.GLM.from_formula('frequency~site*type', family = sm.families.Poisson(), data=df).fit()
    #
    # The saturated model gives a perfect fit, and the fitted data are equal to
    # the original data. Statsmodels indicates a "PerfectSeparationError"

    # Ulcer and aspirin, p. 182 ------------------------------------- 
    inFile = r'GLM_data/Table 9.7 Ulcer and aspirin use.xls'
    df = get_data(inFile)
    df.columns = ['GD', 'CC', 'AP', 'freq']

    model1 = sm.GLM.from_formula('freq~GD+CC+GD*CC', family = sm.families.Poisson(), data=df).fit()
    model2 = sm.GLM.from_formula('freq~GD+CC+GD*CC + AP', family = sm.families.Poisson(), data=df).fit()
    model3 = sm.GLM.from_formula('freq~GD+CC+GD*CC + AP + AP*CC', family = sm.families.Poisson(), data=df).fit()
    model4 = sm.GLM.from_formula('freq~GD+CC+GD*CC + AP + AP*CC + AP*GD', family = sm.families.Poisson(), data=df).fit()
    
    print 'Ulcer and aspirin'
    print model4.fittedvalues


def remission_times_tbd():
    '''Survival analysis / Remission times
    chapter 10.7, p. 201
    These models, also known as "Cox proportional hazards model",
    are currently under development but not yet available in statsmodels.'''

    inFile = r'GLM_data/Table 10.1 Remission times.xls'
    df = get_data(inFile)    
    print df

def longitudinal_data_tbd():
    '''Stroke example
    chapter 11.6, p. 222
    Clustered and Longitudinal Data are described by repeated measures models.
    These are under development, but not yet available in statsmodels.'''

    inFile = r'GLM_data/Table 11.1 Recovery from stroke.xls'
    df = get_data(inFile)    
    print df

if __name__ == '__main__':
    log_linear_models()

