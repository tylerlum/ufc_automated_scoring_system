# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="q2Ink0cSOSRR"
# # UFC Automated Scoring
#
# The goal of this notebook is to:
# * Read in stored, scraped UFC data and convert it into a dataset ready for ML models
# * Train, test, and analyze ML models
#
# Functional as of April 2021

# + [markdown] id="XIv8RUYoOSRW"
# ## Read in stored data

# + id="Ws0PWbZMOSRX"
import numpy as np
import pandas as pd

# + id="3MgHHwyvOSRX"
STORED_FIGHT_TABLE = pd.read_csv('data/April_22_2021_better_data/FIGHT_TABLE_NUM_EVENTS_All_DATA_MODE_Summary_22-04-2021_11:08:22.csv')

# + colab={"base_uri": "https://localhost:8080/", "height": 658} id="PlBKZMo0OSRX" outputId="17b7ba8c-4ed0-467e-bf21-78b4c32eb9b4"
STORED_FIGHT_TABLE

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="7OEfyS34oNMf" outputId="c676e3d6-6af3-4a64-b407-b46603e13981"
STORED_FIGHT_TABLE[(STORED_FIGHT_TABLE["Fighter 0 Total str."] == 0) & (STORED_FIGHT_TABLE["Winner"] == 0)]

# + colab={"base_uri": "https://localhost:8080/", "height": 430} id="5VXivOhCoXa9" outputId="89350981-b76b-4b89-e714-545fb5f3b97b"
STORED_FIGHT_TABLE[(STORED_FIGHT_TABLE["Fighter 1 Total str."] == 0) & (STORED_FIGHT_TABLE["Winner"] == 1)]

# + colab={"base_uri": "https://localhost:8080/", "height": 642} id="uF1ya3BLOSRY" outputId="24a5d200-62aa-4f69-a0d4-a4e2f257ec3b"
# Clean dataset: Only decisions with clear winners
STORED_FIGHT_TABLE = STORED_FIGHT_TABLE[STORED_FIGHT_TABLE["Method"].str.contains("DEC")]
STORED_FIGHT_TABLE = STORED_FIGHT_TABLE[(STORED_FIGHT_TABLE["Winner"] == 1) | (STORED_FIGHT_TABLE["Winner"] == 0)]
STORED_FIGHT_TABLE

# + id="fHjSOBfobBva"
fighter0 = "Robert Whittaker"
fighter1 = "Yoel Romero"

# + id="h2OTM47lT6tq"
controversial = STORED_FIGHT_TABLE[(STORED_FIGHT_TABLE["Fighter 0 Name"] == fighter0) & (STORED_FIGHT_TABLE["Fighter 1 Name"] == fighter1)]
without_controversial = STORED_FIGHT_TABLE.drop(index=controversial.index)

# + colab={"base_uri": "https://localhost:8080/", "height": 147} id="v7UpZyY9UZ70" outputId="54e2c4c9-dd2a-4f0d-b904-0a3b0abe4c6e"
controversial

# + colab={"base_uri": "https://localhost:8080/", "height": 101} id="KrwHq6IaUluO" outputId="aa6a27c6-1097-48f9-caa5-e82ae9e39d48"
without_controversial[(without_controversial["Fighter 0 Name"] == fighter0) & (without_controversial["Fighter 1 Name"] == fighter1)]

# + id="SivCNBMTOSRZ"
X_train = without_controversial.drop(['Winner', 'Fighter 0 Name', 'Fighter 1 Name', 'Method'], axis=1).fillna(0)
y_train = without_controversial[['Winner']]

# + id="YRx9XQSrU8k_"
X_valid = controversial.drop(['Winner', 'Fighter 0 Name', 'Fighter 1 Name', 'Method'], axis=1).fillna(0)
y_valid = controversial[['Winner']]

# + colab={"base_uri": "https://localhost:8080/", "height": 250} id="hPjeCwkvVAMU" outputId="08fc00a8-561d-42b7-8fef-62963920a10d"
X_train.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 196} id="jDjTAPh3VDV7" outputId="e34c9b24-2252-48b1-b1e0-c5971d730122"
y_train.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 130} id="ZXIKtExwVE1K" outputId="addba89c-50a4-47b0-9f7a-4fded538d590"
X_valid.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 77} id="CcWxQryeVGST" outputId="b91601b8-aada-45d5-92ca-314fa69fec92"
y_valid.head()


# + [markdown] id="Jd2Ifk5uOSRa"
# ## Setup train/validate/test split with data augmentation
#
# TODO: Add in smarter data augmentation that create new datapoints nearby.

# + id="I6c8hDK7OSRa"
def create_flipped_table(table):
    '''Rearranges columns of table so that each fight has two rows. Let fighters be A and B.
       One row has (Fighter 0 = A, Fighter 1 = B). One row has (Fighter 0 = B, Fighter 1 = A)
       Ensure same column order, as column names not looked at when passed to ML model'''

    # Get columns in flipped order, which moves the columns around, but changes column name order too
    flipped_columns = []
    for column in table.columns:
        if "Fighter 0" in column:
            flipped_columns.append(column.replace("Fighter 0", "Fighter 1"))
        elif "Fighter 1" in column:
            flipped_columns.append(column.replace("Fighter 1", "Fighter 0"))
        else:
            flipped_columns.append(column)
    flipped_table = table[flipped_columns]

    # Flips winners around
    if 'Winner' in flipped_table.columns:
         flipped_table['Winner'] = flipped_table['Winner'].replace([0, 1], [1, 0])

    # Change column names back to normal
    flipped_table.columns = table.columns
    return flipped_table


def add_rows_of_flipped_columns(table):
    flipped_table = create_flipped_table(table)
    new_table = pd.concat([table, flipped_table])
    return new_table


# + id="0iGtsWJVOSRa"
# Add flipped rows so fighter 0 and 1 are treated same
X_train, y_train = add_rows_of_flipped_columns(X_train), add_rows_of_flipped_columns(y_train)
X_valid, y_valid = add_rows_of_flipped_columns(X_valid), add_rows_of_flipped_columns(y_valid)

# + id="oRU788hJOSRb"
# Expect equal number of examples in Fighter 0 as Fighter 1 from data augmentation
assert(len(y_train[y_train['Winner'] == 0]) == len(y_train[y_train['Winner'] == 1]))
assert(len(y_valid[y_valid['Winner'] == 0]) == len(y_valid[y_valid['Winner'] == 1]))


# + colab={"base_uri": "https://localhost:8080/", "height": 458} id="g208MGkGOSRb" outputId="a492ee92-1b82-4519-8111-89bd8d19a20e"
X_train

# + colab={"base_uri": "https://localhost:8080/", "height": 404} id="wojL_I-7OSRb" outputId="f39e3d02-53c5-4b38-f9e4-7b55cfa1c18f"
y_train

# + colab={"base_uri": "https://localhost:8080/"} id="DndW0X9aOSRc" outputId="16221a60-6f8b-4828-aa8d-1541ba38cc21"
print(f"X_train.shape = {X_train.shape}")
print(f"X_valid.shape = {X_valid.shape}")
print(f"y_train.shape = {y_train.shape}")
print(f"y_valid.shape = {y_valid.shape}")


# + [markdown] id="xNEsuXzFOSRf"
# ### Standardize features and break into fighter 0 and 1

# + id="sKWmgHtTdXLr"
fighter0_columns = [col for col in X_train.columns if "Fighter 0" in col]
fighter1_columns = [col for col in X_train.columns if "Fighter 1" in col]

X0_train = X_train[fighter0_columns]
X1_train = X_train[fighter1_columns]
X0_valid = X_valid[fighter0_columns]
X1_valid = X_valid[fighter1_columns]

X_train_new = pd.concat([X0_train, X1_train], axis=1)
X_valid_new = pd.concat([X0_valid, X1_valid], axis=1) 

means, stds = X_train_new.mean(), X_train_new.std()
X_train_new_normal = (X_train_new - means) / stds
X_valid_new_normal = (X_valid_new - means) / stds

# + [markdown] id="f-sFWdcdiDD1"
# ## Define inputs to future training

# + id="UlvV28UhiAtC"
# X_train, y_train = X_train_new_normal_aug, y_train_aug
X_train = X_train_new_normal
X_valid = X_valid_new_normal

# + [markdown] id="stOorO7tOSRc"
# ## Train and test ML models
#
# TODO: Play around with PyTorch, add in data augmentation like SMOTE, see if normalizing, standardizing, extracting difference features helps. Must be done for deep models. Try out PCA or MDS to visualize.

# + id="1ZgNyaLvVV3G"
import matplotlib.pyplot as plt


# + id="RwGbAJvaOSRc"
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def get_predIdxs_and_trueIdxs(classifier, X, y):
    predIdxs = np.where(classifier.predict(X) > 0.5, 1, 0)
    trueIdxs = y
    return predIdxs, trueIdxs

def plot_confusion_matrix(classifier, X, y):
    predIdxs, trueIdxs = get_predIdxs_and_trueIdxs(classifier, X, y)
    cm = confusion_matrix(trueIdxs, predIdxs)
    cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fighter 0 Win", "Fighter 1 Win"])
    cmDisplay.plot()
    
def print_classification_report(classifier, X, y):
    predIdxs, trueIdxs = get_predIdxs_and_trueIdxs(classifier, X, y)
    print(classification_report(trueIdxs, predIdxs, target_names=["Fighter 0 Win", "Fighter 1 Win"]))


# + [markdown] id="d3v7metxjRlr"
# ### Decision Tree

# + colab={"base_uri": "https://localhost:8080/", "height": 316} id="X0TNR6wajTSX" outputId="783686ff-cb31-43e1-d819-a03abf682f0f"
from sklearn.tree import DecisionTreeClassifier
# Train
decision_tree_clf = DecisionTreeClassifier(random_state=0)
decision_tree_clf.fit(X_train, y_train)

# Validate
accuracy_train = decision_tree_clf.score(X_train, y_train)
accuracy_valid = decision_tree_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# Visualize importances
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, decision_tree_clf.feature_importances_)

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="0NLVskN4jVqU" outputId="8131e0ea-68ef-495b-b2cb-9795719b2a21"
plot_confusion_matrix(decision_tree_clf, X_valid, y_valid)
print_classification_report(decision_tree_clf, X_valid, y_valid)

# + [markdown] id="5jUUa7-6OSRd"
# ### Random forest

# + colab={"base_uri": "https://localhost:8080/", "height": 371} id="B8ly4Z3TOSRd" outputId="2cb833e2-e5b3-4af0-b5bd-2f134f7e59fb"
from sklearn.ensemble import RandomForestClassifier

# Train
random_forest_clf = RandomForestClassifier(max_depth=5, random_state=0)
random_forest_clf.fit(X_train, y_train)

# Validate
accuracy_train = random_forest_clf.score(X_train, y_train)
accuracy_valid = random_forest_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# Visualize importances
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, random_forest_clf.feature_importances_)

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="Zo9BGeJoOSRd" outputId="c0160eb9-5a70-41e7-b4a7-f45ac53c3caa"
plot_confusion_matrix(random_forest_clf, X_valid, y_valid)
print_classification_report(random_forest_clf, X_valid, y_valid)

# + [markdown] id="0613aZAmbHdw"
# ### Extra trees

# + colab={"base_uri": "https://localhost:8080/", "height": 371} id="RTCgZ7rIbHEb" outputId="84787b99-cdae-4225-ea1c-f540e531a5d8"
from sklearn.ensemble import ExtraTreesClassifier

# Train
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
extra_trees_clf.fit(X_train, y_train)

# Validate
accuracy_train = extra_trees_clf.score(X_train, y_train)
accuracy_valid = extra_trees_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# Visualize importances
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, extra_trees_clf.feature_importances_)

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="CRY3quOebU7n" outputId="3209705b-fe3b-4064-81b9-bd80e968722e"
plot_confusion_matrix(extra_trees_clf, X_valid, y_valid)
print_classification_report(extra_trees_clf, X_valid, y_valid)

# + [markdown] id="CXx4w0bQOSRe"
# ### MLP

# + colab={"base_uri": "https://localhost:8080/"} id="cF3J4g8HOSRe" outputId="84c0fa45-2976-4b92-d827-f72b2b3b0d73"
# MLP
from sklearn.neural_network import MLPClassifier

mlp_clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
accuracy_train = mlp_clf.score(X_train, y_train)
accuracy_valid = mlp_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="xsBZgc7wOSRe" outputId="bee1786f-0666-429b-a547-622d5e0009bd"
plot_confusion_matrix(mlp_clf, X_valid, y_valid)
print_classification_report(mlp_clf, X_valid, y_valid)

# + [markdown] id="YaDADd0nOSRe"
# ### SVM

# + colab={"base_uri": "https://localhost:8080/"} id="Lid4_UcGOSRe" outputId="24b07b75-fccc-4655-cd32-3a9265b9dd36"
# SVM
from sklearn.svm import SVC

svm_clf = SVC(random_state=1, probability=True).fit(X_train, y_train)
accuracy_train = svm_clf.score(X_train, y_train)
accuracy_valid = svm_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="9PUOxo-MOSRf" outputId="2bb1b5af-8389-4798-a656-659debe288d0"
plot_confusion_matrix(svm_clf, X_valid, y_valid)
print_classification_report(svm_clf, X_valid, y_valid)

# + colab={"base_uri": "https://localhost:8080/"} id="LzZIqs1OWiXt" outputId="f7fb4c00-c60f-479d-8873-b8bdbc4ffbcc"
# SVM linear kernel
svm_linear_clf = SVC(kernel='linear', random_state=1, probability=True).fit(X_train, y_train)
accuracy_train = svm_linear_clf.score(X_train, y_train)
accuracy_valid = svm_linear_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 281} id="ckmE9q0vXnqs" outputId="091139f1-5dd1-40ae-dc86-744ffe2ed493"
# Visualize importances
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, svm_linear_clf.coef_[0])

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="4iMx96BTWwAY" outputId="99d66a02-e424-4056-d3fb-f3344041b195"
plot_confusion_matrix(svm_linear_clf, X_valid, y_valid)
print_classification_report(svm_linear_clf, X_valid, y_valid)

# + colab={"base_uri": "https://localhost:8080/"} id="7L3PLSKsV-O_" outputId="cc67f5fa-fbd6-4c52-ea87-f0ec3b176ecd"
probability0 = svm_linear_clf.predict_proba(X_valid)[0][0]
probability1 = svm_linear_clf.predict_proba(X_valid)[0][1]
print(f"Probability that {fighter0} won: {probability0}")
print(f"Probability that {fighter1} won: {probability1}")
print(f"Actual winner: {fighter0 if y_valid.iloc[0][0] == 0 else fighter1}")

# + [markdown] id="3itzucb_OSRf"
# ### XGBoost

# + colab={"base_uri": "https://localhost:8080/"} id="H_WSt6GkOSRf" outputId="73930bd7-324d-47f6-a843-8c89fab7df20"
from xgboost import XGBClassifier
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)

accuracy_train = xgb_clf.score(X_train, y_train)
accuracy_valid = xgb_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="KVnq88q6O1Fq" outputId="65147f71-ca1a-42dc-9879-86356814c123"
plot_confusion_matrix(xgb_clf, X_valid, y_valid)
print_classification_report(xgb_clf, X_valid, y_valid)

# + [markdown] id="6wP0UzhhVG4M"
# ### Logistic regression

# + colab={"base_uri": "https://localhost:8080/", "height": 371} id="LbmiFpfCVGgu" outputId="59be9e96-c7cd-4250-acda-a6446e370375"
from sklearn.linear_model import LogisticRegression
logistic_regression_clf = LogisticRegression(random_state=0).fit(X_train, y_train)

accuracy_train = logistic_regression_clf.score(X_train, y_train)
accuracy_valid = logistic_regression_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# Visualize importances
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, logistic_regression_clf.coef_[0])

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="bYiaJ4WKVN0v" outputId="968a52a6-e893-4927-b3c8-53bba650fa57"
plot_confusion_matrix(logistic_regression_clf, X_valid, y_valid)
print_classification_report(logistic_regression_clf, X_valid, y_valid)

# + colab={"base_uri": "https://localhost:8080/", "height": 371} id="cDxUH57MV3CA" outputId="69cc51e8-fde1-4e78-cc0c-00db9b3806be"
logistic_regression_l1_clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=0).fit(X_train, y_train)

accuracy_train = logistic_regression_l1_clf.score(X_train, y_train)
accuracy_valid = logistic_regression_l1_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# Visualize importances
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, logistic_regression_l1_clf.coef_[0])

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="zWOjdd_QV6We" outputId="ae985e76-c9de-4f22-b456-fe8a8b4fda0f"
plot_confusion_matrix(logistic_regression_l1_clf, X_valid, y_valid)
print_classification_report(logistic_regression_l1_clf, X_valid, y_valid)

# + [markdown] id="gzj5WaU_YRsQ"
# ### KNN classifier

# + colab={"base_uri": "https://localhost:8080/"} id="Y5ujO215YRFX" outputId="31af59c1-73a4-4bc3-941b-08c0f7e6327d"
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)

accuracy_train = knn_clf.score(X_train, y_train)
accuracy_valid = knn_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="aMgQlf5rYfSU" outputId="fa312b03-59ab-421e-abca-1cdda4a11256"
plot_confusion_matrix(knn_clf, X_valid, y_valid)
print_classification_report(knn_clf, X_valid, y_valid)

# + [markdown] id="nbCFcybKZgYI"
# ### Gradient boosting

# + colab={"base_uri": "https://localhost:8080/"} id="RQcxuC6PZjMw" outputId="3e63ed9a-9012-4e47-fdea-5277981ef2ac"
from sklearn.ensemble import GradientBoostingClassifier
gradient_boosting_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)

accuracy_train = gradient_boosting_clf.score(X_train, y_train)
accuracy_valid = gradient_boosting_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="uHZNLkgUZo3I" outputId="4054aadf-3216-4d21-e6db-7dadcce81237"
plot_confusion_matrix(gradient_boosting_clf, X_valid, y_valid)
print_classification_report(gradient_boosting_clf, X_valid, y_valid)

# + [markdown] id="FQK1JgIgYwUN"
# ### Adaboost

# + colab={"base_uri": "https://localhost:8080/"} id="8zpZrIACYx0H" outputId="da2db9e2-4afb-4623-ee76-1f4b33acf1cc"
from sklearn.ensemble import AdaBoostClassifier
adaboost_clf = AdaBoostClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)

accuracy_train = adaboost_clf.score(X_train, y_train)
accuracy_valid = adaboost_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="S04qiBcYYzl2" outputId="370b41ed-f131-4724-ad99-4fb9ac88ba29"
plot_confusion_matrix(adaboost_clf, X_valid, y_valid)
print_classification_report(adaboost_clf, X_valid, y_valid)

# + [markdown] id="0MTgv_KmOSRj"
# ### Comparison Model

# + id="CNAmHJFKOSRj"
from tensorflow.keras.layers import Input, Lambda, Subtract, Activation
from tensorflow.keras.models import Model
def create_comparison_model(input_shape):
    num_features_per_fighter = input_shape[0] // 2

    model_ = tf.keras.models.Sequential(name="scoring_deep_model")
    model_.add(tf.keras.Input(shape=num_features_per_fighter))
    model_.add(tf.keras.layers.Dense(32, activation='relu'))
    model_.add(tf.keras.layers.Dropout(0.5))
    model_.add(tf.keras.layers.Dense(16, activation='relu'))
    model_.add(tf.keras.layers.Dropout(0.5))

    model_.add(tf.keras.layers.Dense(1, activation='relu'))
    
    # Run cnn model on each frame
    input_tensor = Input(shape=input_shape, name="input")
    fighter0_state = Lambda(lambda x: x[:, :num_features_per_fighter], name='fighter0_state')(input_tensor)
    fighter1_state = Lambda(lambda x: x[:, num_features_per_fighter:], name='fighter1_state')(input_tensor)

    fighter0_score = model_(fighter0_state)
    fighter1_score = model_(fighter1_state)
    fighter0_score = Lambda(lambda x: x, name='fighter0_score')(fighter0_score)
    fighter1_score = Lambda(lambda x: x, name='fighter1_score')(fighter1_score)
    
    difference_score = Subtract(name='subtracter')([fighter1_score, fighter0_score])
    prediction = Activation('sigmoid', name='sigmoid')(difference_score)
    return Model(inputs=input_tensor, outputs=prediction)


# + colab={"base_uri": "https://localhost:8080/"} id="JVyaVIkzOSRj" outputId="7905b76f-7da8-4105-e5dc-9f25f3be36f7"
comparison_model = create_comparison_model(X_train.shape[1:])
optimizer = tf.keras.optimizers.Adam(lr=0.001)
comparison_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
comparison_model.summary()

# + colab={"base_uri": "https://localhost:8080/", "height": 564} id="teRgJBKjy6wm" outputId="609351ef-0071-455a-8426-40d5d5e1fde7"
tf.keras.utils.plot_model(comparison_model)

# + colab={"base_uri": "https://localhost:8080/"} id="Drp0QsEtOSRk" outputId="66309309-62c8-49ab-ed66-3621bd69ce60"
H = comparison_model.fit(X_train, y_train, epochs=60, validation_data=(X_valid_new_normal, y_valid))

# + colab={"base_uri": "https://localhost:8080/", "height": 277} id="Pc-Uvs2yOSRk" outputId="13a4e7e7-a403-440b-b288-cd3986d1995b"
from matplotlib import pyplot as plt
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy', 'loss', 'val_loss'], loc='upper left')
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="AtM9mTwoOSRk" outputId="561833db-2fc6-404b-8e84-ed02638dc7e9"
plot_confusion_matrix(comparison_model, X_valid_new_normal, y_valid)
print_classification_report(comparison_model, X_valid_new_normal, y_valid)

# + id="tmXQhcXzOSRl"
lo, hi = 0, 1

# + id="B8yB1b1rWAyJ"
X_test_new_normal = X_valid_new_normal

# + id="oZjY_QbBWL6U"
y_test = y_valid

# + colab={"base_uri": "https://localhost:8080/", "height": 130} id="TjyT8mTpThMP" outputId="2d83408d-d83c-42f7-cd60-c6e68f5e2cb8"
X_test_new_normal[lo:hi]

# + id="VA9DyuVbcHGg"
probability = comparison_model.predict(X_test_new_normal[lo:hi])[0]

# + colab={"base_uri": "https://localhost:8080/"} id="D24siDVIOSRl" outputId="01ec1ba1-bfc5-4397-883c-114a23295f40"
comparison_model.predict(X_test_new_normal[lo:hi])

# + colab={"base_uri": "https://localhost:8080/", "height": 77} id="MK6NhCojOSRl" outputId="43a2a7fb-82b5-486a-aeb9-4ad71ad255e7"
y_test[lo:hi]

# + colab={"base_uri": "https://localhost:8080/"} id="icRxd_bKOSRm" outputId="78138e52-e55f-498a-9388-5020a6748074"
subtracter = comparison_model.get_layer('subtracter').output
subtracter = Model(comparison_model.input, subtracter)
subtracter.predict(X_test_new_normal[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="pp_GHFfGOSRm" outputId="8e93e4c8-bfc3-4a94-b258-94b934c5a7b0"
fighter0_score = comparison_model.get_layer('fighter0_score').output
fighter0_score = Model(comparison_model.input, fighter0_score)
fighter0_score.predict(X_test_new_normal[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="sRbpamFhOSRm" outputId="f23fd2fc-51e1-4fc9-a00e-daaef1e83ef5"
fighter1_score = comparison_model.get_layer('fighter1_score').output
fighter1_score = Model(comparison_model.input, fighter1_score)
fighter1_score.predict(X_test_new_normal[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="bOrJ8nxvcddj" outputId="49ab52ee-c798-4ed1-c809-cb5bda001eb8"
score0 = fighter0_score.predict(X_test_new_normal[lo:hi])[0][0]
score1 = fighter1_score.predict(X_test_new_normal[lo:hi])[0][0]
prediction = comparison_model.predict(X_test_new_normal[lo:hi])[0][0]
print(f"{fighter0} score: {score0}")
print(f"{fighter1} score: {score1}")
print(f"Probability that {fighter1} won: {prediction}")
print(f"Actual winner: {fighter0 if y_test.iloc[0][0] == 0 else fighter1}")

# + colab={"base_uri": "https://localhost:8080/", "height": 147} id="75PEDBGhRpo1" outputId="87c6b49f-5804-4794-d73e-e3e6400882e2"
controversial

# + colab={"base_uri": "https://localhost:8080/", "height": 160} id="c4g8g3OFRxWe" outputId="c5ca4282-0960-414b-8006-539d0b6a4b16"
X_test_new_normal

# + colab={"base_uri": "https://localhost:8080/"} id="zHa31HYgOSRn" outputId="ad063b90-f754-489b-eca9-194f525e94c7"
# 
columns = list(X_test_new_normal.columns)
new_columns = columns[len(columns)//2:] + columns[:len(columns)//2]
switcheroo = X_test_new_normal[new_columns]
fighter1_score.predict(switcheroo[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="_OvDdtBjOSRn" outputId="3e3a6e4a-59d5-4820-c106-8b96357e8163"
fighter0_score.predict(switcheroo[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="975q2avhOSRn" outputId="6850cdb1-2ab1-4cf0-a3cd-9d8367f961c0"
subtracter.predict(switcheroo[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="uci1y0klSyUk" outputId="4bbfcfa6-ec1a-47e4-c144-0799cddf0cd8"
comparison_model.predict(switcheroo[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="_daq4cdDS2Fw" outputId="567b4fa8-0d97-4f27-c17a-2198ae602879"
deep_model.predict(X_test_new_normal[1:10])

# + colab={"base_uri": "https://localhost:8080/"} id="QjNmdPPhS-TB" outputId="bd49edaf-4411-4c2d-8555-1c2e6301e574"
deep_model.predict(switcheroo[1:10])

# + id="SI88RDxQS_7I"

