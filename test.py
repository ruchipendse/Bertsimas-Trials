import numpy as np
import pandas as pd
from sklearn import metrics 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ad_df = pd.read_csv('add.csv', index_col=0,low_memory=False)

ad_df2 = ad_df.applymap(lambda val: np.nan if str(val).strip() == '?' else val)
ad_df2 = ad_df2.dropna()


lb = LabelEncoder()
y = lb.fit_transform(ad_df2.iloc[:, -1])

X = ad_df2.iloc[:,:-1]


sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))