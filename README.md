# EXNO:4-DS
## Date: 17/10/2025
## NAME: ANISH ADAN THIVAKARAN
## REF.NO: 25017997
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

### FEATURE SCALING:
```
import pandas as pd
df=pd.read_csv("bmi.csv")
df
```

<img width="411" height="510" alt="Screenshot 2025-10-16 193544" src="https://github.com/user-attachments/assets/ce90406f-9972-493e-af31-8844bb17ed72" />

```
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,Normalizer,RobustScaler
df1=df.copy()
enc=StandardScaler()
df1[['new height','new weight']]=enc.fit_transform(df[['Height','Weight']])
df1
```

<img width="566" height="525" alt="Screenshot 2025-10-16 193554" src="https://github.com/user-attachments/assets/5be987fc-ff26-4739-a387-82ab8d2866b1" />

```
df2=df.copy()
enc=MinMaxScaler()
df2[['new height','new weight']]=enc.fit_transform(df[['Height','Weight']])
df2
```

<img width="551" height="520" alt="Screenshot 2025-10-16 193607" src="https://github.com/user-attachments/assets/b63d1fde-ff61-4a0b-95f6-9a683f49cc27" />

```
df3=df.copy()
enc=MaxAbsScaler()
df3[['new height','new weight']]=enc.fit_transform(df[['Height','Weight']])
df3
```

<img width="544" height="521" alt="Screenshot 2025-10-16 193620" src="https://github.com/user-attachments/assets/a9cd8844-e1cd-47cd-8092-337e9b144426" />

```
df4=df.copy()
enc=RobustScaler()
df4[['new height','new weight']]=enc.fit_transform(df[['Height','Weight']])
df4
```

<img width="560" height="530" alt="Screenshot 2025-10-16 193633" src="https://github.com/user-attachments/assets/516fce7e-815e-4fc7-8cb8-3d3192452f4c" />

```
df5=df.copy()
enc=Normalizer()
df5[['new height','new weight']]=enc.fit_transform(df[['Height','Weight']])
df5
```

<img width="568" height="526" alt="Screenshot 2025-10-16 193644" src="https://github.com/user-attachments/assets/5cb13cec-e851-4279-8bf8-b95a43d9c023" />

### FEATURE SELECTION:

```
import pandas as pd
df=pd.read_csv("income(1).csv")
df
```

<img width="1089" height="738" alt="Screenshot 2025-10-16 212014" src="https://github.com/user-attachments/assets/578873c3-8b02-4df1-a8c0-904d207171a1" />


```
from sklearn.preprocessing import LabelEncoder

df_encoded=df.copy()
le=LabelEncoder()
for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col]=le.fit_transform(df_encoded[col])

x=df_encoded.drop("SalStat",axis=1)
y=df_encoded["SalStat"]
x
```

<img width="934" height="410" alt="Screenshot 2025-10-16 212026" src="https://github.com/user-attachments/assets/34633952-de2e-4c4b-9c22-44b467592f92" />


```
y
```

<img width="345" height="219" alt="Screenshot 2025-10-16 212038" src="https://github.com/user-attachments/assets/4923f192-f7d1-400c-9aa5-f8a5df3ae76a" />


```
from sklearn.feature_selection import SelectKBest, chi2
chi2_selector=SelectKBest(chi2,k=5)
chi2_selector.fit(x,y)
selected_features_chi2=x.columns[chi2_selector.get_support()]
print("Selected features (Chi-Square):",list(selected_features_chi2))

mi_scores=pd.Series(chi2_selector.scores_,index=x.columns)
print(mi_scores.sort_values(ascending=False))
```

<img width="759" height="247" alt="Screenshot 2025-10-16 212052" src="https://github.com/user-attachments/assets/a8fa9f92-2389-4bbb-8d5c-e5b99b38726c" />


```
from sklearn.feature_selection import f_classif
anova_selector=SelectKBest(f_classif,k=5)
anova_selector.fit(x,y)
selected_features_anova=x.columns[anova_selector.get_support()]
print("Selected features (ANOVA F-test):",list(selected_features_anova))

mi_scores=pd.Series(anova_selector.scores_,index=x.columns)
print(mi_scores.sort_values(ascending=False))
```

<img width="733" height="255" alt="Screenshot 2025-10-16 212101" src="https://github.com/user-attachments/assets/aaaff26e-dcca-4819-be49-cc208d06474b" />


```
from sklearn.feature_selection import mutual_info_classif

mi_selector=SelectKBest(mutual_info_classif,k=5)
mi_selector.fit(x,y)

selected_features_mi=x.columns[mi_selector.get_support()]
print("Selected features (Mutual Info):",list(selected_features_mi))

mi_scores=pd.Series(mi_selector.scores_,index=x.columns)
print("\nMutual Informtion Scores:\n",mi_scores.sort_values(ascending=False))
```

<img width="728" height="284" alt="Screenshot 2025-10-16 212115" src="https://github.com/user-attachments/assets/f3b45cec-80c8-48e0-ad63-c68c60cf45d2" />


```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model=LogisticRegression(max_iter=100)
rfe=RFE(model,n_features_to_select=5)
rfe.fit(x,y)

selected_features_rfe=x.columns[rfe.support_]
print("Selected features (RFE):",list(selected_features_rfe))
```

<img width="679" height="652" alt="Screenshot 2025-10-16 212234" src="https://github.com/user-attachments/assets/997644f1-9be9-4bf2-a5af-7e4ad5ea4cc5" />


```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

model = LogisticRegression(max_iter=200)
sfs = SequentialFeatureSelector(model, n_features_to_select=5, direction='forward',n_jobs=-1)
sfs.fit(x, y)
selected_features = x.columns[sfs.get_support()]
print(list(selected_features))
```

<img width="704" height="39" alt="Screenshot 2025-10-16 212305" src="https://github.com/user-attachments/assets/c2cf95d5-a907-4c73-81e9-fec37831a5d2" />


```
from sklearn.linear_model import LassoCV
import numpy as np
lasso =LassoCV(cv=5).fit(x, y)
importance = np.abs(lasso.coef_)
print(importance)
selected_features_lasso=x.columns[importance > 0]
print("Selected features (Lasso):", list(selected_features_lasso))
```

<img width="742" height="107" alt="Screenshot 2025-10-16 212315" src="https://github.com/user-attachments/assets/e3011a80-ecd8-4ca6-868a-b47785648c51" />


```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("income(1).csv")
le=LabelEncoder()
df_encoded=df.copy()
for col in df_encoded.select_dtypes (include="object").columns:
    df_encoded [col]=le.fit_transform(df_encoded[col])

x=df_encoded.drop("SalStat", axis=1) 
y = df_encoded ["SalStat"]

x_train, x_test, y_train, y_test=train_test_split(
    x, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

<img width="530" height="346" alt="Screenshot 2025-10-16 212326" src="https://github.com/user-attachments/assets/3a69c59a-be3f-4d17-8e38-b61d1f263b4f" />


# RESULT:
Thus, the program to implement Feature Scaling and Feature Selection was completed successfully.
