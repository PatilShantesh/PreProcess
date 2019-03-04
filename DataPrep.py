from __future__ import division
import os
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import graphviz
import matplotlib.pyplot as plt
import statistics
from sklearn.ensemble import RandomForestClassifier
import random 
import statistics
from statsmodels import robust
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from statsmodels.tools.tools import add_constant
import operator


# Create an output folder to store all process files with in the current working directory
def createOutputFolder(str):
    try:    
        directoryPath=os.getcwd()+"\\"+str
        if not os.path.exists(directoryPath):
            os.makedirs(directoryPath)
        return "Success"
    except Exception as e:
        print ('\n\n Error in Creating Output folder : \n'+str(e))

        
#Function to read CSV Data
#Input  : File name with complete path
#Output : A Data frame
def readFromcsv(FileName):
    try:
        data=pd.read_csv(FileName, sep=',',header=0)
        return data
    except Exception as e:
        print ('\n\n Error in reading data from '+FileName+' : \n'+str(e))
        return pd.DataFrame()
    
    
# Function to find column level NaN occurance
# It calculates the % of NaN occurance in each columns
# Input  : Data Frame
# Output : Nan Summary of all columns which have atleast one NaN entry.
def getcolumnsNanDetails(dataFrame):
    try:
        missingValueSummaryDF=pd.DataFrame()
        if dataFrame.shape[0]>0:
            numberOfColumns=dataFrame.shape[0]
            missingValueSummaryDF=pd.DataFrame(dataFrame.isnull().sum(axis = 0))
            missingValueSummaryDF.reset_index(level=0, inplace=True)
            missingValueSummaryDF.columns=["columnName","NaNCount"]
            missingValueSummaryDF['NaNRatio']=(missingValueSummaryDF['NaNCount']/numberOfColumns)*100
            missingValueSummaryDF=missingValueSummaryDF[missingValueSummaryDF.NaNCount>0]
            missingValueSummaryDF.sort_values(["NaNCount","columnName"], ascending=[False, True], inplace=True)
            return missingValueSummaryDF
        else:
            print ('\n\n No Data found to calculate getcolumnsNanDetails')
    except Exception as e:
        print ('\n\nError in getcolumnsNanDetails Calculation \n'+str(e))
    return missingValueSummaryDF

    
# Function to find rows which has more than 'NaNThreshold' number of NaN columns.
# Input  : DataFrame    : Dataframe to find rows
#          NaNThreshold : Number of columns to be considered (threashold)           
# Output : Data frame with index name and number of missing column in the corresponding row
def getRowsWithMoreNaNs(dataFrame,NaNThreshold):
    try:
        missingValueSummaryDF=pd.DataFrame()
        if dataFrame.shape[0]>0:
            numberOfColumns=dataFrame.shape[1]
            missingValueSummaryDF=pd.DataFrame(dataFrame.isnull().sum(axis = 1))
            missingValueSummaryDF.reset_index(level=0, inplace=True)
            missingValueSummaryDF.columns=["RowIndex","NaNCount"]
            missingValueSummaryDF['NaNRatio']=(missingValueSummaryDF['NaNCount']/numberOfColumns)*100
            missingValueSummaryDF=missingValueSummaryDF[missingValueSummaryDF["NaNRatio"]>=NaNThreshold]
            missingValueSummaryDF=missingValueSummaryDF[["NaNCount","RowIndex","NaNRatio"]]
            missingValueSummaryDF.sort_values(["NaNCount","RowIndex"], ascending=[False, True], inplace=True)
            missingValueSummaryDF=missingValueSummaryDF[["RowIndex","NaNCount","NaNRatio"]]
        else:
            print ('\n\n No Data found to calculate getRowsWithMoreNaNs')
    except Exception as e:
        print ('\n\nError in getRowsWithMoreNaNs Calculation \n'+str(e))
    return missingValueSummaryDF

# Function to impute null values with passed value
# Input  : col : List with null values
#          val : Value to be imputed with
# Output : list with null values are inputed with mode items
def imputeNull(col,val):
    try:
        col.fillna(val, inplace=True)
    except Exception as e:
        print ('\n\nError in imputeWithModeVal Calculation \n'+str(e))
    return col

def plotROC_ReturnAUC(y_train,train_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_train, train_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw=2
    plt.plot(fpr,tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return round(roc_auc,2)
   
## function will return position of outliers
def detect_outlier_medianzScore(data_1,threshold=3):
    outlierDF=pd.DataFrame()
    outliers=[]
    outliersIndex=[]
    zScore=[]
    
    medianVal = np.median(data_1)
    MAD =np.median([np.abs(x - medianVal) for x in data_1]) 
    
    cnt=0
    for y in data_1:
        z_score= np.abs(y - medianVal)/MAD 
        if z_score > threshold:
            outliersIndex.append(cnt)
            outliers.append(y)
            zScore.append(z_score)
        cnt=cnt+1
        
    outlierDF['Index']=outliersIndex
    outlierDF['OutlierValue']=outliers
    outlierDF['median']=medianVal
    outlierDF['MAD']=MAD
    outlierDF['medianzScore']=zScore
    return outlierDF

def detect_outlier_meanzScore(data_1,threshold=3):
    outlierDF=pd.DataFrame()
    outliers=[]
    outliersIndex=[]
    zScore=[]
    
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)  
    
    cnt=0
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliersIndex.append(cnt)
            outliers.append(y)
        cnt=cnt+1        
    outlierDF['Index']=outliersIndex
    outlierDF['OutlierValue']=outliers
    outlierDF['medianzScore']=zScore
    outlierDF['mean']=np.mean(data_1)
    outlierDF['std']=np.std(data_1)  
    return outlierDF


def plotRangeBinHist(df,columnName,numOfBins):
    
    minVal=min(df[columnName])
    maxVal=max(df[columnName])
    dataBins=np.linspace(minVal, maxVal, numOfBins)
    out= pd.cut(df[columnName], bins=dataBins, include_lowest=True)
    out= out.value_counts().reindex(out.cat.categories)
    out=pd.DataFrame(out)
    
    
    x_labels=list(out.index)
    frequencies=list(out[columnName])
    freq_series = pd.Series.from_array(frequencies)

    #x_labels = [108300.0, 110540.0, 112780.0, 115020.0, 117260.0, 119500.0,
    #            121740.0, 123980.0, 126220.0, 128460.0, 130700.0]

    # Plot the figure.
    plt.figure(figsize=(12, 8))
    ax = freq_series.plot(kind='bar')
    ax.set_title(columnName+' Frequency')
    ax.set_xlabel(columnName)
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(x_labels)


    def add_value_labels(ax, spacing=3):
        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.1f}".format(y_value)

            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va)                      # Vertically align label differently for
                                            # positive and negative values.


    # Call the function above. All the magic happens there.
    add_value_labels(ax)

    plt.savefig("image.png")

    
def getCorrtoRemove(corrDF,pos,cutOff):
    toRemove=[]
    for j in range(0,corrDF.shape[0]):
        if pos!=j and (corrDF.iloc[pos,j]>cutOff or corrDF.iloc[pos,j]< cutOff*-1):
            toRemove.append(corrDF.columns[j])
    return toRemove            


def removeCorr(dataSet,cutOff=.99) : 
    corrDF=dataSet.corr()
    colLen=corrDF.shape[0]
    corrDF.to_csv("corrDFbefore.csv",sep=",")
    for i in range(0,dataSet.shape[1]):
        if i<corrDF.shape[1]:
            removeList=getCorrtoRemove(corrDF,i,cutOff)
            dataSet=dataSet.drop(removeList, axis=1)
            corrDF=dataSet.corr()
        else:
            corrDF.to_csv("corrDFafter.csv",sep=",")
            return dataSet
    corrDF.to_csv("corrDFafter.csv",sep=",")
    return dataSet



def calculate_vif_(df, thresh=5):
    '''
    Calculates VIF each feature in a pandas dataframe
    A constant must be added to variance_inflation_factor or the results will be incorrect

    :param X: the pandas dataframe
    :param thresh: the max VIF value before the feature is removed from the dataframe
    :return: dataframe with features removed
    '''
    const = add_constant(df)
    cols = const.columns
    variables = np.arange(const.shape[1])
    vif_df = pd.Series([variance_inflation_factor(const.values, i) 
               for i in range(const.shape[1])], 
              index=const.columns).to_frame()

    vif_df = vif_df.sort_values(by=0, ascending=False).rename(columns={0: 'VIF'})
    vif_df = vif_df.drop('const')
    vif_df = vif_df[vif_df['VIF'] > thresh]

    print ('Features above VIF threshold:\n')
    print (vif_df[vif_df['VIF'] > thresh])

    col_to_drop = list(vif_df.index)

    for i in col_to_drop:
        print ('Dropping: {}'.format(i))
        df = df.drop(columns=i)
    return df


#Function will generate number of bins and its frequesncy summary
def generateBins(df,columnName,numOfBins):  
    colLen=df[columnName].dropna().count()
    minVal=df[columnName].dropna().min()
    maxVal=df[columnName].dropna().max()
    #print minVal,maxVal,numOfBins
    dataBins=np.linspace(minVal, maxVal, numOfBins)
    #print dataBins
    out= pd.cut(df[df[columnName].notnull()][columnName], bins=dataBins, include_lowest=True)
    out= out.value_counts().reindex(out.cat.categories)
    out=pd.DataFrame(out)
    out.columns=['Count']
    out['overallPerc']=(out['Count']/colLen)*100
    out['BinRange']=out.index
    out.sort_values(['Count'], ascending=[False], inplace=True)
    out=out[['BinRange','overallPerc','Count']]
    return out

def getOutlierRange(df,cutOff):
    percVal=0
    cnt=0
    for i in range(0,df.shape[0]-1):
        if percVal<=cutOff:
            percVal=percVal+df.iloc[i]['overallPerc']
            cnt=cnt+1
    df=df.iloc[0:cnt:]
    return df