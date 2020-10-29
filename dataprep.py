import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer, IterativeImputer
from collections import Counter


class DataTest:

    def __init__(self):
        self.data_csv = self.nancheck = self.data_imputed = self.file = None
        self.nancols = []
        self.cat_cols = []
        self.num_cols = []
        self.datetime_cols = []
        self.other_cols = []

    def importdata(self, file):
        self.file = file
        self.data_csv = pd.read_csv(self.file)
        return self.data_csv

    def dropduplicates(self):
        self.data_csv.drop_duplicates(inplace=True)
        return self.data_csv

    def dropconstants(self):
        feats_counts = self.data_csv.nunique(dropna=False)
        constant_features = feats_counts.loc[feats_counts == 1].index.tolist()
        self.data_csv.drop(constant_features, axis=1, inplace=True)
        return self.data_csv

        # check for unique values and dtypes
        # unique_counts=self.data_csv.from_records([(col,df[col].nunique()) for col in df.columns],
        # columns=["Column_Name","Num_Unique"]).sort_values(by=["Num_Unique"])
        # print(unique_counts)

    def sepcols(self):
        for col in self.data_csv.columns:
            if self.data_csv[col].dtype == "object":
                self.cat_cols.append(col)
            elif self.data_csv[col].dtype == "float64" or self.data_csv[col].dtype == "int64":
                self.num_cols.append(col)
            elif self.data_csv[col].dtype == "datetime64":
                self.datetime_cols.append(col)
            else:
                self.other_cols.append(col)
        return self.cat_cols, self.num_cols, self.datetime_cols, self.other_cols

    def dropoutliers(self):
        if self.num_cols == []:
            try:
                for col in self.data_csv.columns:
                    if self.data_csv[col].dtype == "float64" or self.data_csv[col].dtype == "int64":
                        self.num_cols.append(col)
                a = self.num_cols[0]
            except IndexError:
                raise ValueError(
                    "Data have no numeric column therefore you cant find any outliers."
                    "Check your dtypes.")

        outlier_idx = []
        for each in self.num_cols:
            q3 = np.percentile(self.data_csv[each], 75)
            q1 = np.percentile(self.data_csv[each], 25)
            iqr = q3 - q1
            step = iqr * 1.5
            maxm = q3 + step
            minm = q1 - step
            outlier_list = self.data_csv[(self.data_csv[each] < minm) | (self.data_csv[each] > maxm)].index
            outlier_idx.extend(outlier_list)
        outlier_idx = Counter(outlier_idx)
        multiple_outliers = list(i for i, v in outlier_idx.items() if v >= 1)
        
        self.data_csv = self.data_csv.drop(multiple_outliers, axis=0).reset_index(drop=True)
        return self.data_csv

    def checknans(self):
        self.nancheck = self.data_csv.isnull().sum()
        for each in self.nancheck.index:
            if self.nancheck.loc[each] >= 1:
                self.nancols.append(each)

    def fillnans(self, method=None, value=None):
        self.data_imputed = self.data_csv.copy(deep=True)
        if method == "impute":
            for each in self.nancols:
                if each in self.cat_cols:
                    self.fillcats()
                elif each in self.num_cols:
                    self.fillnums()
                elif each in self.datetime_cols:
                    self.filltimes()
                else:
                    self.fillother()
        elif method == "fill" and value == 0:
            self.data_imputed.fillna(0, inplace=True)
        elif method == "fill" and value == 1:
            self.data_imputed.fillna(1, inplace=True)
        elif method is None:
            raise ValueError(
                "Method cannot be empty.")
        elif method == "fill" and value is None:
            raise ValueError(
                "You must specify the value to fill.")
        elif method == "fill" and value != 0 or value != 1:     
            raise ValueError(
                "Value must be 0 or 1.")
        return self.data_imputed

    def scaling(self, method=None):
        scaler = MinMaxScaler()
        ss = StandardScaler()
        if method == "minmax":
            self.data_imputed[self.num_cols] = scaler.fit_transform(self.data_imputed[self.num_cols])
        elif method == "standard":
            self.data_imputed[self.num_cols] = ss.fit_transform(self.data_imputed[self.num_cols])
        return self.data_imputed

    def fillcats(self):
        knn_imputer = KNNImputer()
        enc_dict = {}
        for col_name in self.cat_cols:
            enc_dict[col_name] = OrdinalEncoder()

            col = self.data_csv[col_name]
            col_not_null = col[col.notnull()]
            reshaped_vals = col_not_null.values.reshape(-1, 1)
            encoded_vals = enc_dict[col_name].fit_transform(reshaped_vals)
            self.data_csv.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)
        self.data_imputed.loc[:, self.cat_cols] = np.round(
            knn_imputer.fit_transform(self.data_csv.loc[:, self.cat_cols]))
        for col in self.cat_cols:
            reshaped = self.data_imputed[col].values.reshape(-1, 1)
            self.data_imputed[col] = enc_dict[col].inverse_transform(reshaped)

    def fillnums(self):
        imputer = IterativeImputer()
        self.data_imputed.loc[:, self.num_cols] = np.round(
            imputer.fit_transform(self.data_imputed.loc[:, self.num_cols]))

    def filltimes(self):
        self.data_imputed.loc[:, self.datetime_cols].interpolate(method="quadratic", inplace=True)

    def fillother(self):
        self.data_imputed.loc[:, self.other_cols].fillna(method="ffill", inplace=True)

    def save(self):
        self.data_imputed.to_csv("data_imputed.csv", index=False)
        print("Saved.")

    def start(self, file):
        self.importdata(file)
        self.dropduplicates()
        self.dropconstants()
        self.sepcols()
        self.dropoutliers()
        self.checknans()
        self.fillnans(method="fill", value=0)
        self.scaling()
        self.save()

        result = self.data_imputed
        print("done")
        return result
