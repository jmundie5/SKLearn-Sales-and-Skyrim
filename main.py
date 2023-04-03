import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
#sparse=False
enc = OneHotEncoder(handle_unknown='ignore',sparse=False)
pd.set_option("display.max.columns", 80)
pd.set_option("display.width", 240)
#-----------------------VAULTING DATA-------------------------------
SkryimSetWhole = pd.read_csv("NEWSKYRIMSET.csv")
SkryimSet = SkryimSetWhole.iloc[:778,:]
SkyrimVault = SkryimSetWhole.iloc[778:,:]
SkyrimVault.to_csv("skryimVault.csv",index=None)
SkryimSet.to_csv("SkryimSet.csv",index=None)
#--------------------------FIXING DATA------------------------------
SkryimSet = SkryimSet.set_index('Name')

enc_df = pd.DataFrame(enc.fit_transform(SkryimSet[['Gender']]))

SkryimSet = SkryimSet.reset_index()
SkryimSet = SkryimSet.join(enc_df)
SkryimSet = SkryimSet.set_index('Name')
SkryimSet.rename({
    0: 'Female',
    1: 'Animal',
    2: 'Male',
    3: 'Radient NPC'
},
         axis=1,
         inplace=True)
SkryimSet = SkryimSet.drop(['Gender'], axis=1)
#---------------------------LOGISTIC SKYRIM-----------------
print("-----------SKYRIM DATASET-----------")
drop_these = ['Race']

X = SkryimSet.drop(drop_these,axis=1).to_numpy().reshape(len(SkryimSet),-1)
Y = SkryimSet['Race'].to_numpy()


Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y, shuffle=True)
lr = LogisticRegression(max_iter=100000)
lr.fit(Xtrain,ytrain)
print("Logistic Target is the NPC's Race")
print(f"Your Skryim NPC Race Predictor (Training) score is : {lr.score(Xtrain,ytrain)}")
print(f"Your Skryim NPC Race Predictor (Test) score is : {lr.score(Xtest,ytest)}")
print("")
print("Confusion Matrix")
print(confusion_matrix(ytest,lr.predict(Xtest)))
print("")

#------------------------LINEAR SKYRIM---------------------------------
drop_these = ['Level (PC=10)',"Race"]
X = SkryimSet.drop(drop_these,axis=1).to_numpy().reshape(len(SkryimSet),-1)
Y = SkryimSet['Level (PC=10)'].to_numpy()
Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y, shuffle=True)
lr = LinearRegression()
lr.fit(Xtrain,ytrain)
print("Linear Target is the NPC's Level")
print(f"Your Skryim NPC Level Predictor (Training) score is : {lr.score(Xtrain,ytrain)}")
print(f"Your Skryim NPC Level Predictor (Test) score is : {lr.score(Xtest,ytest)}")


#result1 = SkryimSet.head(5)
#print("--------------------------------------------------------")
#print(result1)
#print("--------------------------------------------------------")
#------------------------------------------------------------------
#------------------------VIDEOGAME SALES---------------------------
#------------------------------------------------------------------
#LOGISTIC TARGET : GAME DEVELOPER 
#LINEAR TARGET : GLOBAL SALES
#--------------------VAULTING DATA-----------------------
print("-------------------------------------------------------------------------------------------------------------------------")
print("-----------VIDEO GAME SALES DATASET-----------")
VGSalesWhole = pd.read_csv("VIDEOGAMESALES.csv")
VGSalesWhole=VGSalesWhole.dropna(axis=0)
VGSalesSet = VGSalesWhole.iloc[:8000,:]
VGSalesVault = VGSalesWhole.iloc[8000:,:]
VGSalesVault.to_csv("VGSalesVault.csv",index=None)
VGSalesSet.to_csv("VGSalesSet.csv",index=None)
#--------------------------FIXING DATA------------------------------
VGSalesSet = VGSalesSet.set_index('Name')
enc = OneHotEncoder(handle_unknown='ignore',sparse=False)
enc_df = pd.DataFrame (enc.fit_transform(VGSalesSet[['Platform']]))
enc_df.columns = enc.get_feature_names_out(['Platform'])
VGSalesSet.drop(['Platform'] ,axis=1, inplace=True)
VGSalesSet = VGSalesSet.reset_index()
VGSalesSet= pd.concat([VGSalesSet, enc_df ], axis=1)
VGSalesSet = VGSalesSet.set_index('Name')

enc = OneHotEncoder(handle_unknown='ignore',sparse=False)
enc_df = pd.DataFrame (enc.fit_transform(VGSalesSet[['Genre']]))
enc_df.columns = enc.get_feature_names_out(['Genre'])
VGSalesSet.drop(['Genre'] ,axis=1, inplace=True)
VGSalesSet = VGSalesSet.reset_index()
VGSalesSet= pd.concat([VGSalesSet, enc_df ], axis=1)
VGSalesSet = VGSalesSet.set_index('Name')
#---------------------------LOGICAL VIDEO GAME SALES--------------------------
#------------------------------TARGET : DEVELOPER --------------------------
drop_these = ["Publisher"]

X = np.c_[VGSalesSet.NA_Sales,VGSalesSet.EU_Sales,VGSalesSet.JP_Sales]
#VGSalesSet.drop(drop_these,axis=1).to_numpy().reshape(len(VGSalesSet),-1)
Y = VGSalesSet["Publisher"].to_numpy()


Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y, shuffle=True)
lr = LogisticRegression(max_iter=100000)
lr.fit(Xtrain,ytrain)
print("Logistic Target is the Videogame Publisher")
print(f"Your Videogame Publisher Predictor (Training) score is : {lr.score(Xtrain,ytrain)}")
print(f"Your Videogame Publisher Predictor (Test) score is : {lr.score(Xtest,ytest)}")

#np.set_printoptions(threshold=np.inf)



print("")
print("Confusion Matrix")
print("(Line 120 includes commented code to print entire matrix, takes much longer)")
print(confusion_matrix(ytest,lr.predict(Xtest)))
print("")

#---------------------------LINEAR VIDEO GAME SALES--------------------------

drop_these = ['Global_Sales',"Publisher"]
X = VGSalesSet.drop(drop_these,axis=1).to_numpy().reshape(len(VGSalesSet),-1)
Y = VGSalesSet['Global_Sales'].to_numpy()
Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y, shuffle=True)
lr = LinearRegression()
lr.fit(Xtrain,ytrain)
print("Linear Target is the Global Sales in Millions")
print(f"Your Videogame Global Sale Predictor (Training) score is : {lr.score(Xtrain,ytrain)}")
print(f"Your Videogame Global Sale Predictor (Test) score is : {lr.score(Xtest,ytest)}")

#result1 = VGSalesSet.head(5)
#print("--------------------------------------------------------")
#print(result1)
#print("--------------------------------------------------------")