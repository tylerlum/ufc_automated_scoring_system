# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/tylerlum/ufc_automated_scoring_system/blob/main/UFC_automated_scoring.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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
STORED_FIGHT_TABLE = pd.read_csv('FIGHT_TABLE_NUM_EVENTS_All_DATA_MODE_Summary_22-04-2021_11:08:22.csv')

# + colab={"base_uri": "https://localhost:8080/", "height": 657} id="PlBKZMo0OSRX" outputId="c3d7efa1-5a1e-48a3-8ef8-e860e43c023e"
STORED_FIGHT_TABLE

# + colab={"base_uri": "https://localhost:8080/", "height": 640} id="uF1ya3BLOSRY" outputId="437ecbfd-a5d6-42ad-e64b-c60d930d8c23"
# Clean dataset: Only decisions with clear winners
STORED_FIGHT_TABLE = STORED_FIGHT_TABLE[STORED_FIGHT_TABLE["Method"].str.contains("DEC")]
STORED_FIGHT_TABLE = STORED_FIGHT_TABLE[(STORED_FIGHT_TABLE["Winner"] == 1) | (STORED_FIGHT_TABLE["Winner"] == 0)]
STORED_FIGHT_TABLE

# + id="SivCNBMTOSRZ"
X = STORED_FIGHT_TABLE.drop(['Winner', 'Fighter 0 Name', 'Fighter 1 Name', 'Method'], axis=1).fillna(0)
y = STORED_FIGHT_TABLE[['Winner']]

# + colab={"base_uri": "https://localhost:8080/", "height": 455} id="ePWHHGDiOSRZ" outputId="c211382d-fdd4-498c-eaef-449702015273"
X

# + colab={"base_uri": "https://localhost:8080/", "height": 402} id="-13NNIUoOSRZ" outputId="83aeddc0-e9ef-447e-aed2-703493cf7b6c"
y


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
# Train/validate/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=0)

# Add flipped rows so fighter 0 and 1 are treated same
X_train, y_train = add_rows_of_flipped_columns(X_train), add_rows_of_flipped_columns(y_train)
X_valid, y_valid = add_rows_of_flipped_columns(X_valid), add_rows_of_flipped_columns(y_valid)
X_test, y_test = add_rows_of_flipped_columns(X_test), add_rows_of_flipped_columns(y_test)

# + id="oRU788hJOSRb"
# Expect equal number of examples in Fighter 0 as Fighter 1 from data augmentation
assert(len(y_train[y_train['Winner'] == 0]) == len(y_train[y_train['Winner'] == 1]))
assert(len(y_valid[y_valid['Winner'] == 0]) == len(y_valid[y_valid['Winner'] == 1]))
assert(len(y_test[y_test['Winner'] == 0]) == len(y_test[y_test['Winner'] == 1]))

# + colab={"base_uri": "https://localhost:8080/", "height": 455} id="g208MGkGOSRb" outputId="8d7f37ae-722e-4285-f255-de544ae009fd"
X_train

# + colab={"base_uri": "https://localhost:8080/", "height": 402} id="wojL_I-7OSRb" outputId="ab800758-1a0d-40fc-dde0-5ed1820d6b39"
y_train

# + colab={"base_uri": "https://localhost:8080/"} id="DndW0X9aOSRc" outputId="2ee4f6d8-e3a0-4dfa-8b49-c81b511307ab"
print(f"X_train.shape = {X_train.shape}")
print(f"X_valid.shape = {X_valid.shape}")
print(f"X_test.shape = {X_test.shape}")
print(f"y_train.shape = {y_train.shape}")
print(f"y_valid.shape = {y_valid.shape}")
print(f"y_test.shape = {y_test.shape}")

# + [markdown] id="xNEsuXzFOSRf"
# ### Standardize features and break into fighter 0 and 1

# + id="sKWmgHtTdXLr"
fighter0_columns = [col for col in X_train.columns if "Fighter 0" in col]
fighter1_columns = [col for col in X_train.columns if "Fighter 1" in col]

X0_train = X_train[fighter0_columns]
X1_train = X_train[fighter1_columns]
X0_valid = X_valid[fighter0_columns]
X1_valid = X_valid[fighter1_columns]
X0_test = X_test[fighter0_columns]
X1_test = X_test[fighter1_columns]

X_train_new = pd.concat([X0_train, X1_train], axis=1)
X_valid_new = pd.concat([X0_valid, X1_valid], axis=1) 
X_test_new = pd.concat([X0_test, X1_test], axis=1)

means, stds = X_train_new.mean(), X_train_new.std()
X_train_new_normal = (X_train_new - means) / stds
X_valid_new_normal = (X_valid_new - means) / stds
X_test_new_normal = (X_test_new - means) / stds

# + colab={"base_uri": "https://localhost:8080/"} id="x3836SoTgnx8" outputId="3f2daefc-9add-4897-98c3-79640a9f101d"
# Add data augmentation only on training data (can try SMOTE, gaussian noise, etc)
extra_train_copies = 10
mu, sigma = 0, 0.1
noisy_copies = [X_train_new_normal + np.random.normal(mu, sigma, X_train_new_normal.shape) for _ in range(extra_train_copies)]
print(f"X_train_new_normal.shape = {X_train_new_normal.shape}")
print(f"y_train.shape = {y_train.shape}")
X_train_new_normal_aug = pd.concat([X_train_new_normal] + noisy_copies, axis=0)
y_train_aug = pd.concat([y_train] + [y_train] * extra_train_copies, axis=0)
print(f"X_train_new_normal_aug.shape = {X_train_new_normal_aug.shape}")
print(f"y_train_aug.shape = {y_train_aug.shape}")

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

# + [markdown] id="asxsbETjkIrk"
# ### PCA

# + colab={"base_uri": "https://localhost:8080/", "height": 326} id="fESHfkdUkIWm" outputId="a25fb340-9203-4b34-9fa8-1ef8e5930b68"
from sklearn.decomposition import PCA

# Fit PCA
pca = PCA(n_components=2)
Z_train = pca.fit_transform(X_train)
print(f"pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}")

# Plot
color_dict = {0: 'red', 1: 'blue'}
for label in np.unique(y_train):
    ix = np.where(y_train == label)
    plt.scatter(Z_train[ix, 0], Z_train[ix, 1], c = color_dict[label], label = label, s = 100)
plt.legend()
plt.title("PCA")
plt.xlabel("Component 0")
plt.ylabel("Component 1")

# + colab={"base_uri": "https://localhost:8080/", "height": 296} id="Jw6qNhormm4a" outputId="83df8419-9d0a-4b6f-e39b-8cbe890c7b1f"
# Understand basis
component0 = pca.components_[0, :]
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, component0)
plt.title("Component 0")

# + colab={"base_uri": "https://localhost:8080/", "height": 296} id="4CXfHZnzmoZw" outputId="4a5da31d-ca4a-4152-dc3e-d93eade2f87d"
# Understand basis
component1 = pca.components_[1, :]
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, component1)
plt.title("Component 1")

# + [markdown] id="nwHwpx3JnZEX"
# ### Sparse PCA

# + colab={"base_uri": "https://localhost:8080/", "height": 309} id="2chZbqHxndt_" outputId="65569fd2-f698-41cf-dba2-c59ffc3cb6fe"
from sklearn.decomposition import SparsePCA

# Fit SparsePCA
sparse_pca = SparsePCA(n_components=2)
Z_train = sparse_pca.fit_transform(X_train)

# Plot
color_dict = {0: 'red', 1: 'blue'}
for label in np.unique(y_train):
    ix = np.where(y_train == label)
    plt.scatter(Z_train[ix, 0], Z_train[ix, 1], c = color_dict[label], label = label, s = 100)
plt.legend()
plt.title("Sparse PCA")
plt.xlabel("Component 0")
plt.ylabel("Component 1")

# + colab={"base_uri": "https://localhost:8080/", "height": 296} id="EVAhzpjqn3YJ" outputId="67c2bc8f-e0c1-485d-f2f2-12e2f45ff287"
# Understand basis
component0 = sparse_pca.components_[0, :]
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, component0)
plt.title("Component 0")

# + colab={"base_uri": "https://localhost:8080/", "height": 296} id="4NKPlOXGn-38" outputId="3cb8d4d9-4a7d-4cd3-d37e-789b28a547d2"
# Understand basis
component1 = sparse_pca.components_[1, :]
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, component1)
plt.title("Component 1")

# + [markdown] id="9363WETbnYAz"
# ### NMF

# + colab={"base_uri": "https://localhost:8080/", "height": 309} id="JuNbh2L8oIuG" outputId="6b992e52-79d4-486a-dcea-05fa71eb6df8"
from sklearn.decomposition import NMF

# Fit NMF
nmf = NMF(n_components=2, init='random', random_state=0)
if (X_train < 0).values.any():  # Must be non-negative
    Z_train = nmf.fit_transform(X_train - X_train.min())
else:
    Z_train = nmf.fit_transform(X_train)

# Plot
color_dict = {0: 'red', 1: 'blue'}
for label in np.unique(y_train):
    ix = np.where(y_train == label)
    plt.scatter(Z_train[ix, 0], Z_train[ix, 1], c = color_dict[label], label = label, s = 100)
plt.legend()
plt.title("NMF")
plt.xlabel("Component 0")
plt.ylabel("Component 1")

# + colab={"base_uri": "https://localhost:8080/", "height": 296} id="uUAETqfcoM-9" outputId="a5c1143f-6a08-474a-9367-7848574b0818"
# Understand basis
component0 = nmf.components_[0, :]
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, component0)
plt.title("Component 0")

# + colab={"base_uri": "https://localhost:8080/", "height": 296} id="KQPRyVvxnH8W" outputId="41517782-48cf-4885-8da4-d5a1d8ab0bbf"
# Understand basis
component1 = nmf.components_[1, :]
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, component1)
plt.title("Component 1")

# + [markdown] id="EBi_AVwWnG7B"
# ### MDS

# + colab={"base_uri": "https://localhost:8080/", "height": 309} id="dR3d8dHhovBU" outputId="d8d14d51-9c27-4592-d353-b44b6263a622"
from sklearn.manifold import MDS

# Fit MDS
mds = MDS(n_components=2, random_state=0)
Z_train = mds.fit_transform(X_train)

# Plot
color_dict = {0: 'red', 1: 'blue'}
for label in np.unique(y_train):
    ix = np.where(y_train == label)
    plt.scatter(Z_train[ix, 0], Z_train[ix, 1], c = color_dict[label], label = label, s = 100)
plt.legend()
plt.title("MDS")
plt.xlabel("Component 0")
plt.ylabel("Component 1")

# + [markdown] id="Xja9LLuWpIEm"
# ### TSNE

# + colab={"base_uri": "https://localhost:8080/", "height": 309} id="YozL-pVPpKC1" outputId="be10520c-9857-482d-beae-add568472669"
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)
Z_train = tsne.fit_transform(X_train)

# Plot
color_dict = {0: 'red', 1: 'blue'}
for label in np.unique(y_train):
    ix = np.where(y_train == label)
    plt.scatter(Z_train[ix, 0], Z_train[ix, 1], c = color_dict[label], label = label, s = 100)
plt.legend()
plt.title("TSNE")
plt.xlabel("Component 0")
plt.ylabel("Component 1")

# + [markdown] id="unt_aWe1nEtd"
# ### Helper functions for evaluation

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

# + colab={"base_uri": "https://localhost:8080/", "height": 317} id="X0TNR6wajTSX" outputId="2f1a0b08-8ebc-4686-e0a0-a04c61aeb8db"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="0NLVskN4jVqU" outputId="62547603-d819-4026-e998-651088399237"
plot_confusion_matrix(decision_tree_clf, X_valid, y_valid)
print_classification_report(decision_tree_clf, X_valid, y_valid)

# + [markdown] id="5jUUa7-6OSRd"
# ### Random forest

# + colab={"base_uri": "https://localhost:8080/", "height": 372} id="B8ly4Z3TOSRd" outputId="e8e76404-a03c-4f0d-da26-6ff6621e70b3"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="Zo9BGeJoOSRd" outputId="3982c4c8-ecde-40ce-df77-e4e1a39dff96"
plot_confusion_matrix(random_forest_clf, X_valid, y_valid)
print_classification_report(random_forest_clf, X_valid, y_valid)

# + [markdown] id="0613aZAmbHdw"
# ### Extra trees

# + colab={"base_uri": "https://localhost:8080/", "height": 372} id="RTCgZ7rIbHEb" outputId="7214851f-8bb9-4c94-ab40-09bf72089dd8"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="CRY3quOebU7n" outputId="13743cd6-9021-470e-fdb5-0390930775fc"
plot_confusion_matrix(extra_trees_clf, X_valid, y_valid)
print_classification_report(extra_trees_clf, X_valid, y_valid)

# + [markdown] id="CXx4w0bQOSRe"
# ### MLP

# + colab={"base_uri": "https://localhost:8080/"} id="cF3J4g8HOSRe" outputId="8790d309-c811-467d-d7af-2ac13510001b"
# MLP
from sklearn.neural_network import MLPClassifier

mlp_clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
accuracy_train = mlp_clf.score(X_train, y_train)
accuracy_valid = mlp_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="xsBZgc7wOSRe" outputId="f925a930-fd35-42f4-d35a-ecdbd587c720"
plot_confusion_matrix(mlp_clf, X_valid, y_valid)
print_classification_report(mlp_clf, X_valid, y_valid)

# + [markdown] id="YaDADd0nOSRe"
# ### SVM

# + colab={"base_uri": "https://localhost:8080/"} id="Lid4_UcGOSRe" outputId="7b0ef01b-01ae-4e6f-e838-64bc7aba7baa"
# SVM
from sklearn.svm import SVC

svm_clf = SVC(random_state=1).fit(X_train, y_train)
accuracy_train = svm_clf.score(X_train, y_train)
accuracy_valid = svm_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="9PUOxo-MOSRf" outputId="e3a25d07-a08c-4f3f-d045-c51775bddbda"
plot_confusion_matrix(svm_clf, X_valid, y_valid)
print_classification_report(svm_clf, X_valid, y_valid)

# + colab={"base_uri": "https://localhost:8080/"} id="LzZIqs1OWiXt" outputId="95631fec-67ed-4083-9fe5-d1dcf8dc2721"
# SVM linear kernel
svm_linear_clf = SVC(kernel='linear', random_state=1).fit(X_train, y_train)
accuracy_train = svm_linear_clf.score(X_train, y_train)
accuracy_valid = svm_linear_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 282} id="ckmE9q0vXnqs" outputId="56b6236d-2561-4244-b14f-e8b789b19ed7"
# Visualize importances
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, svm_linear_clf.coef_[0])

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="4iMx96BTWwAY" outputId="89411ed0-69c5-49d0-f79e-770cfe9b9676"
plot_confusion_matrix(svm_linear_clf, X_valid, y_valid)
print_classification_report(svm_linear_clf, X_valid, y_valid)

# + [markdown] id="3itzucb_OSRf"
# ### XGBoost

# + colab={"base_uri": "https://localhost:8080/"} id="H_WSt6GkOSRf" outputId="ca5a2c2a-3dc5-4803-bde4-657f1e07d591"
from xgboost import XGBClassifier
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)

accuracy_train = xgb_clf.score(X_train, y_train)
accuracy_valid = xgb_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 438} id="KVnq88q6O1Fq" outputId="07ca8b46-69c2-4d8a-9e4e-b7e29776c4a2"
plot_confusion_matrix(xgb_clf, X_valid, y_valid)
print_classification_report(xgb_clf, X_valid, y_valid)

# + [markdown] id="6wP0UzhhVG4M"
# ### Logistic regression

# + colab={"base_uri": "https://localhost:8080/", "height": 372} id="LbmiFpfCVGgu" outputId="29bea944-ab40-4ce5-b0bb-bdd25ff216e6"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="bYiaJ4WKVN0v" outputId="e2dc2ba1-5e80-4e36-cd47-fb20c1810201"
plot_confusion_matrix(logistic_regression_clf, X_valid, y_valid)
print_classification_report(logistic_regression_clf, X_valid, y_valid)

# + colab={"base_uri": "https://localhost:8080/", "height": 372} id="cDxUH57MV3CA" outputId="7fe6b233-e0d5-4396-eb57-f7b7ed5824e0"
logistic_regression_l1_clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=0).fit(X_train, y_train)

accuracy_train = logistic_regression_l1_clf.score(X_train, y_train)
accuracy_valid = logistic_regression_l1_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# Visualize importances
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, logistic_regression_l1_clf.coef_[0])

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="zWOjdd_QV6We" outputId="4b19e7b6-3d7e-4a07-f79b-ce35338c74d9"
plot_confusion_matrix(logistic_regression_l1_clf, X_valid, y_valid)
print_classification_report(logistic_regression_l1_clf, X_valid, y_valid)

# + [markdown] id="gzj5WaU_YRsQ"
# ### KNN classifier

# + colab={"base_uri": "https://localhost:8080/"} id="Y5ujO215YRFX" outputId="377dcfa7-84f4-4163-ec34-cf37f80078b2"
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)

accuracy_train = knn_clf.score(X_train, y_train)
accuracy_valid = knn_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="aMgQlf5rYfSU" outputId="43a54041-83c0-42ac-d5e8-5bcb9fa1b41b"
plot_confusion_matrix(knn_clf, X_valid, y_valid)
print_classification_report(knn_clf, X_valid, y_valid)

# + [markdown] id="nbCFcybKZgYI"
# ### Gradient boosting

# + colab={"base_uri": "https://localhost:8080/"} id="RQcxuC6PZjMw" outputId="ab0371d2-e017-4544-c795-c2852e56f5b9"
from sklearn.ensemble import GradientBoostingClassifier
gradient_boosting_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)

accuracy_train = gradient_boosting_clf.score(X_train, y_train)
accuracy_valid = gradient_boosting_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="uHZNLkgUZo3I" outputId="9255ab4c-b70a-45b1-9cd7-f4abccc1ed9b"
plot_confusion_matrix(gradient_boosting_clf, X_valid, y_valid)
print_classification_report(gradient_boosting_clf, X_valid, y_valid)

# + [markdown] id="FQK1JgIgYwUN"
# ### Adaboost

# + colab={"base_uri": "https://localhost:8080/"} id="8zpZrIACYx0H" outputId="6ccc27e5-b065-4bb4-a773-4cd92a69c9f4"
from sklearn.ensemble import AdaBoostClassifier
adaboost_clf = AdaBoostClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)

accuracy_train = adaboost_clf.score(X_train, y_train)
accuracy_valid = adaboost_clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="S04qiBcYYzl2" outputId="4f06571d-8c0e-4d04-c575-132b06e5fa86"
plot_confusion_matrix(adaboost_clf, X_valid, y_valid)
print_classification_report(adaboost_clf, X_valid, y_valid)

# + [markdown] id="ksOlSZIgOSRg"
# ### Deep model

# + colab={"base_uri": "https://localhost:8080/"} id="Ao5ESVIgOSRh" outputId="88f0960d-6a1a-4bed-f635-61fb90649ff0"
# FFN
import tensorflow as tf

deep_model = tf.keras.models.Sequential()
deep_model.add(tf.keras.Input(shape=X_train.shape[1:]))
deep_model.add(tf.keras.layers.Dense(16, activation='relu'))
deep_model.add(tf.keras.layers.Dense(16, activation='relu'))
deep_model.add(tf.keras.layers.Dropout(0.5))
deep_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

deep_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
deep_model.summary()

# + colab={"base_uri": "https://localhost:8080/"} id="Pe43H-DXOSRh" outputId="7937fe16-131c-4be2-f2e2-2a6440ad6d0f"
H = deep_model.fit(X_train, y_train, epochs=100, validation_data=(X_valid_new_normal, y_valid))

# + colab={"base_uri": "https://localhost:8080/", "height": 277} id="acYFP63oOSRh" outputId="7fdb3c03-f002-4bc1-d09f-a88a857d2b44"
from matplotlib import pyplot as plt
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy', 'loss', 'val_loss'], loc='upper left')
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="hrDpuSjjOSRi" outputId="85b851f5-46cf-4284-c571-2000f2ae6b3b"
plot_confusion_matrix(deep_model, X_valid_new_normal, y_valid)
print_classification_report(deep_model, X_valid_new_normal, y_valid)

# + [markdown] id="0MTgv_KmOSRj"
# ### Comparison Model

# + id="CNAmHJFKOSRj"
from tensorflow.keras.layers import Input, Lambda, Subtract, Activation
from tensorflow.keras.models import Model
def create_comparison_model(input_shape):
    num_features_per_fighter = input_shape[0] // 2

    model_ = tf.keras.models.Sequential()
    model_.add(tf.keras.Input(shape=num_features_per_fighter))
    model_.add(tf.keras.layers.Dense(32, activation='relu'))
    model_.add(tf.keras.layers.Dense(32, activation='relu'))
    model_.add(tf.keras.layers.Dropout(0.5))

    model_.add(tf.keras.layers.Dense(1, activation='relu'))
    
    # Run cnn model on each frame
    input_tensor = Input(shape=input_shape)
    fighter0_state = Lambda(lambda x: x[:, :num_features_per_fighter], name='fighter0_state')(input_tensor)
    fighter1_state = Lambda(lambda x: x[:, num_features_per_fighter:], name='fighter1_state')(input_tensor)

    fighter0_score = model_(fighter0_state)
    fighter1_score = model_(fighter1_state)
    fighter0_score = Lambda(lambda x: x, name='fighter0_score')(fighter0_score)
    fighter1_score = Lambda(lambda x: x, name='fighter1_score')(fighter1_score)
    
    difference_score = Subtract(name='subtracter')([fighter1_score, fighter0_score])
    prediction = Activation('sigmoid')(difference_score)
    return Model(inputs=input_tensor, outputs=prediction)


# + colab={"base_uri": "https://localhost:8080/"} id="JVyaVIkzOSRj" outputId="0536e687-0170-4994-a83e-0889665f4de8"
comparison_model = create_comparison_model(X_train.shape[1:])
optimizer = tf.keras.optimizers.Adam(lr=0.001)
comparison_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
comparison_model.summary()

# + colab={"base_uri": "https://localhost:8080/"} id="Drp0QsEtOSRk" outputId="ef976487-05cf-40de-8a48-4ee8b286aacd"
H = comparison_model.fit(X_train, y_train, epochs=100, validation_data=(X_valid_new_normal, y_valid))

# + colab={"base_uri": "https://localhost:8080/", "height": 277} id="Pc-Uvs2yOSRk" outputId="00082320-4758-48f5-a98a-70f2c312b6e4"
from matplotlib import pyplot as plt
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy', 'loss', 'val_loss'], loc='upper left')
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="AtM9mTwoOSRk" outputId="f45f7c2a-65fb-4cb4-8ec4-6327c4fc1654"
plot_confusion_matrix(comparison_model, X_valid_new_normal, y_valid)
print_classification_report(comparison_model, X_valid_new_normal, y_valid)

# + id="tmXQhcXzOSRl"
lo, hi = 11, 20

# + colab={"base_uri": "https://localhost:8080/", "height": 367} id="m_W-6PQpOSRl" outputId="77547f5a-9400-4afa-8109-3b64e85c779f"
X_test_new[lo:hi]

# + colab={"base_uri": "https://localhost:8080/", "height": 367} id="TjyT8mTpThMP" outputId="069add08-8f86-4af9-83e3-5324efcb93af"
X_test_new_normal[lo:hi]

# + colab={"base_uri": "https://localhost:8080/"} id="D24siDVIOSRl" outputId="109e761a-befe-4a5b-c2bd-c486f2b49000"
comparison_model.predict(X_test_new_normal[lo:hi])

# + colab={"base_uri": "https://localhost:8080/", "height": 314} id="MK6NhCojOSRl" outputId="dcb6dd57-a702-4d07-8cbd-8b94d7caa46d"
y_test[lo:hi]

# + colab={"base_uri": "https://localhost:8080/"} id="icRxd_bKOSRm" outputId="36b20726-937f-4bdf-81c3-3cce1dea586f"
subtracter = comparison_model.get_layer('subtracter').output
subtracter = Model(comparison_model.input, subtracter)
subtracter.predict(X_test_new_normal[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="pp_GHFfGOSRm" outputId="be6ffcca-2b86-468f-f7d1-73601e54265f"
fighter0_score = comparison_model.get_layer('fighter0_score').output
fighter0_score = Model(comparison_model.input, fighter0_score)
fighter0_score.predict(X_test_new_normal[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="sRbpamFhOSRm" outputId="ff946bad-c414-4dbe-9cf2-e195a0bea6af"
fighter1_score = comparison_model.get_layer('fighter1_score').output
fighter1_score = Model(comparison_model.input, fighter1_score)
fighter1_score.predict(X_test_new_normal[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="zHa31HYgOSRn" outputId="1ddc49e6-f1af-43f9-9297-89c36c59e646"
columns = list(X_test_new_normal.columns)
new_columns = columns[len(columns)//2:] + columns[:len(columns)//2]
switcheroo = X_test_new_normal[new_columns]
fighter1_score.predict(switcheroo[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="_OvDdtBjOSRn" outputId="6bba6447-04c5-4667-f355-817dc9c77e5e"
fighter0_score.predict(switcheroo[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="975q2avhOSRn" outputId="82cb0d3c-da57-4ccd-8a2a-64873d683fbd"
subtracter.predict(switcheroo[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="uci1y0klSyUk" outputId="d6742f2a-8fc0-4a72-a3bd-3f27ca6ebe2d"
comparison_model.predict(switcheroo[lo:hi])

# + colab={"base_uri": "https://localhost:8080/"} id="_daq4cdDS2Fw" outputId="a4fc7a7a-4e61-4c8c-9468-27e12d03d9b2"
deep_model.predict(X_test_new_normal[1:10])

# + colab={"base_uri": "https://localhost:8080/"} id="QjNmdPPhS-TB" outputId="833606fc-ac00-4bb2-916f-9ab978b26828"
deep_model.predict(switcheroo[1:10])

# + id="SI88RDxQS_7I"

