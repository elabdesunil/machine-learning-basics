# Machine Learning Basics

## Machine Learing involves the following steps:

1. Import the Data - `we usually work csv`
2. Clean the Data - `removing or modifying duplicates, irrelevant or incomplete. if data is text based such as name of countries or gender need to be converted to numerical values`
3. Split the Data into Training/Test Sets - `split into training and test sets to check the accuracy of the data, ex. 80% of the data for training and 20% for testing`
4. Create a Model - `involves selecting an algorithm to analize the data. ex. Decision Tree, Neural Networks etc`
5. Train the Model - `feeding the training data to the model to look for patterns in the data`
6. Make Predictions - `give an input and ask the model to make predictions`
7. Evaluate and Improve - `evaluate the prediction, measure the accuracy, then either select a different algorithm that might produces more accurate result or fine tune the parameters of the model`

## Libraries used:

- Pandas
- Scikit-Learn

## Prediction using DecisionTreeClassifier

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
music_data = pd.read_csv('music.csv')
music_data.head()

X = music_data.drop(columns=['genre'])
X.head()

y = music_data['genre']
y.head()

model = DecisionTreeClassifier()
model.fit(X,y)

predictions = model.predict([[21, 1], [22, 0]])
predictions
```

## Training and Testing with DecisionTreeClassifier, train_test_split and accuracy_score

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')

X = music_data.drop(columns=['genre'])
y = music_data['genre']
# X_train, X_test are input values for training and testing
# y_train, y_test are output values for training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# test_size 0.2 means 20% of data is allocated for testing
# the more the test size is, the less will be the training data =>> means less accuracy

model = DecisionTreeClassifier()
model.fit(X_train, y_train) # only passing training data sets
predictions = model.predict(X_test) # X_test input values for testing

score = accuracy_score(y_test, predictions) # comparing y_test with the predictions we got from the model.predict
score # accuracy score of 1.0 means 100%
```

## Saving Machine Learning Model with joblib (`Note: joblib has been deprecated`)

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
# joblib object has methods for loading and saving models

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)

# predictions = model.predict([[21, 1]])

joblib.dump(model, 'music-recommender.joblib')

loaded_model = joblib.load('music-recommender.joblib')

prediction_joblib = loaded_model.predict([[21, 1]])
prediction_joblib
```

## Generating a GraphViz representation of the decision tree with export_graphviz

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns = ['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)

tree.export_graphviz(model,
                    out_file='music-recommender.dot',
                    feature_names = ['age', 'gender'],
                    class_names= sorted(y.unique()),
                    label = 'all',
                    rounded=True,
                    filled=True)
```
