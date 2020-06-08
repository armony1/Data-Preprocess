import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import KNNImputer, IterativeImputer
from collections import Counter

class DataTest:
    
    def __init__(self):

        pass
        
    def importdata(self):
        
        self.data_as_csv = pd.read_csv(self.file)
        
        return self
    
        # Açıklama eklenecek.
    def dropduplicates(self):
        
        self.data_as_csv.drop_duplicates()
        
        return self
    
    def dropconstants(self):
        
        feats_counts = self.data_as_csv.nunique(dropna = False)
        feats_counts.sort_values()
        constant_features = feats_counts.loc[feats_counts == 1].index.tolist()
        self.data_as_csv.drop(constant_features, axis = 1, inplace = True)
        
        return self
    
        #check for unique values and dtypes
        
        #unique_counts=self.data_as_csv.from_records([(col,df[col].nunique()) for col in df.columns],columns=["Column_Name","Num_Unique"]).sort_values(by=["Num_Unique"])
        #print(unique_counts)
        
    def sepcols(self):
        
        self.cat_cols = []
        self.num_cols = []
        self.datetime_cols = []
        self.other_cols = []
        
        for col in self.data_as_csv.columns:
            
            if self.data_as_csv[col].dtype == "object":
                self.cat_cols.append(col)
            elif self.data_as_csv[col].dtype == "float64" or self.data_as_csv[col].dtype == "int64":
                self.num_cols.append(col)
            elif self.data_as_csv[col].dtype == "datetime64":
                self.datetime_cols.append(col)
                
            else:
                self.other_cols.append(col)
                
        return self
    
    def dropoutliers(self):
        
        outlier_idx = []

        for each in self.num_cols:
            Q3 = np.percentile(self.data_as_csv[each], 75)
            Q1 = np.percentile(self.data_as_csv[each], 25)
            IQR = Q3 - Q1
            step = IQR * 1.5
    
            maxm = Q3 + step
            minm = Q1 - step
    
            outlier_list = self.data_as_csv[(self.data_as_csv[each] < minm) | (self.data_as_csv[each] > maxm)].index
            outlier_idx.extend(outlier_list)
    
        outlier_idx = Counter(outlier_idx)
        multiple_outliers = list(i for i, v in outlier_idx.items() if v >= 1)

        self.data_as_csv = self.data_as_csv.drop(multiple_outliers,axis = 0).reset_index(drop = True)
        
        return self
    
    def checknans(self):
        
        self.nancheck = self.data_as_csv.isnull().sum()
        self.nancols = []
        for each in self.nancheck.index:
            if self.nancheck.loc[each] >= 1:
                self.nancols.append(each)

        return self
    
    def fillnans(self):
        
        self.data_imputed = self.data_as_csv.copy(deep = True)

        for each in self.nancols:
            
            if each in self.cat_cols:
                self.fillcats()
            elif each in self.num_cols:
                self.fillnums()
            elif each in self.datetime_cols:
                self.filltimes()
            else:
                
                self.fillother()
        
        return self
    
    def scaling(self):
        
        scaler = MinMaxScaler()

        self.data_imputed[self.num_cols] = scaler.fit_transform(self.data_imputed[self.num_cols])
        
        return self
    
    def fillcats(self):
        
        KNN_imputer = KNNImputer()
        enc_dict = {}
        
        for col_name in self.cat_cols:
            enc_dict[col_name] = OrdinalEncoder()
            
            col = self.data_as_csv[col_name]
            col_not_null = col[col.notnull()]
            reshaped_vals = col_not_null.values.reshape(-1, 1)
            
            encoded_vals = enc_dict[col_name].fit_transform(reshaped_vals)
            self.data_as_csv.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)
        
                
        self.data_imputed.loc[:, self.cat_cols] = np.round(KNN_imputer.fit_transform(self.data_as_csv.loc[:, self.cat_cols]))
        
        for col in self.cat_cols:
            reshaped = self.data_imputed[col].values.reshape(-1, 1)
            self.data_imputed[col] = enc_dict[col].inverse_transform(reshaped)
        
        return self
    
    def fillnums(self):
        
        imputer = IterativeImputer()
        self.data_imputed.loc[:, self.num_cols] = np.round(imputer.fit_transform(self.data_imputed.loc[:, self.num_cols]))

        return self
    
    def filltimes(self):
        
        self.data_imputed.loc[:, self.datetime_cols].interpolate(method = "quadratic", inplace = True)
        
        return self
    
    def fillother(self):
        
        self.data_imputed.loc[:, self.other_cols].fillna(method = "ffill", inplace = True)
        
        return self
    
    def save(self):
        
        self.data_imputed.to_csv("data_imputed.csv", index = False)
        print("Saved.")
        
        return self
    
    def start(self, file):
        
        self.file = file
        
        self.importdata()
        self.dropduplicates()
        self.dropconstants()
        self.sepcols()
        self.dropoutliers()
        self.checknans()
        self.fillnans()
        self.scaling()
        self.save()
        
        print("done")
        
        return self.data_imputed
