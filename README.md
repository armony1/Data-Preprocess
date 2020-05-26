# Data-Preprocess

I've made a simple script that automates datepreprocessing for datasets. It includes:

1-) Drops Duplicates

2-) Drops Constants

3-) Seperates columns and checks for outliers in numerical columns and drops them

4-) Chechs NaN values in both categorical and numerical columns(also other columns like datetime) and fills NaN values

   4a-) If missing values are in categorical features first turns them to a numerical feature and fills it with KNN imputer
   
   4b-) If missing values are numerical features it fills them with sklearn's iterative imputer
   
   4c-) If missing values are datetime columns fills them with interpolation
   
   4d-) If none of the above uses forward fill
    
5-) Then apply min max scaling to numerical columns and export to a file called data_imputed.csv

Notes:

If feature dtypes are wrong script will likely to be fail, i did not add any complex features to fix dtypes. So before you use it make sure that your dtpes are correct.

