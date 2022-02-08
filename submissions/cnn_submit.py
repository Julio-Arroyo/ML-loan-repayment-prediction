import numpy as np
import xgboost as xgb
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels
import category_encoders as ce
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn


train_data = pd.read_csv('../data/LOANS_TRAIN.csv')
test_data = pd.read_csv('../data/LOANS_TEST.csv')

id_column = test_data['id']

train_data.drop(columns=['id','grade', 'emp_title', 'title', 'earliest_cr_line', 'issue_d', 'zip_code'], axis=1, inplace=True)
test_data.drop(columns=['id','grade', 'emp_title', 'title', 'earliest_cr_line', 'issue_d', 'zip_code'], axis=1, inplace=True)

# we want to ultimately use this data, but its nominal multi-categorical nature requires further preprocessing
# prolly with OneHotEncoder
    # Deal with nominal, multi-categorical data
    # Goal: convert home_ownership (RENT, OWN, MORTGAGE, OTHER) to a 4D vector.
        # if MORTGAGE ==> [0,0,1,0]
# print(len(train_data['zip_code'].unique()))
# print(train_data['zip_code'].unique())
ce_OHE = ce.OneHotEncoder(cols=['addr_state', 'home_ownership', 'purpose'])
print('BEFORE')
print(train_data.shape)
train_data = ce_OHE.fit_transform(train_data)
print('AFTER')
print(train_data.shape)
test_data = ce_OHE.fit_transform(test_data)
# NOTE: 'purpose' only has 14 categories, probably good things to learn from

# there are no joint applications in training, so model won't be able to learn it. Better to drop it
train_data.drop(columns=['application_type', 'purpose_14'], axis=1, inplace=True)
test_data.drop(columns=['application_type'], axis=1, inplace=True)


labelencoder = LabelEncoder()
labelencoder2 = LabelEncoder()
# Assigning numerical values and storing in another column
train_data['sub_grade'] = labelencoder.fit_transform(train_data['sub_grade'])
train_data['emp_length'].replace('< 1 year', 0, inplace=True)
train_data['emp_length'].replace('1 year', 1.0, inplace=True)
train_data['emp_length'].replace('2 years', 2.0, inplace=True)
train_data['emp_length'].replace('3 years', 3.0, inplace=True)
train_data['emp_length'].replace('4 years', 4.0, inplace=True)
train_data['emp_length'].replace('5 years', 5.0, inplace=True)
train_data['emp_length'].replace('6 years', 6.0, inplace=True)
train_data['emp_length'].replace('7 years', 7.0, inplace=True)
train_data['emp_length'].replace('8 years', 8.0, inplace=True)
train_data['emp_length'].replace('9 years', 9.0, inplace=True)
train_data['emp_length'].replace('10 years', 10.0, inplace=True)
train_data['emp_length'].replace('10+ years', 15.0, inplace=True)
train_data['emp_length'] = train_data['emp_length'].fillna(0)
train_data['pub_rec_bankruptcies'] = train_data['pub_rec_bankruptcies'].fillna(0)
train_data['verification_status'].replace('Verified', 1, inplace=True)
train_data['verification_status'].replace('Source Verified', 1, inplace=True)
train_data['verification_status'].replace('Not Verified', 0, inplace=True)
train_data['initial_list_status'].replace('w', 1, inplace=True)
train_data['initial_list_status'].replace('f', 0, inplace=True)

test_data['sub_grade'] = labelencoder2.fit_transform(test_data['sub_grade'])
test_data['emp_length'].replace('< 1 year', 0, inplace=True)
test_data['emp_length'].replace('1 year', 1.0, inplace=True)
test_data['emp_length'].replace('2 years', 2.0, inplace=True)
test_data['emp_length'].replace('3 years', 3.0, inplace=True)
test_data['emp_length'].replace('4 years', 4.0, inplace=True)
test_data['emp_length'].replace('5 years', 5.0, inplace=True)
test_data['emp_length'].replace('6 years', 6.0, inplace=True)
test_data['emp_length'].replace('7 years', 7.0, inplace=True)
test_data['emp_length'].replace('8 years', 8.0, inplace=True)
test_data['emp_length'].replace('9 years', 9.0, inplace=True)
test_data['emp_length'].replace('10 years', 10.0, inplace=True)
test_data['emp_length'].replace('10+ years', 15.0, inplace=True)
test_data['emp_length'] = test_data['emp_length'].fillna(0)
test_data['pub_rec_bankruptcies'] = test_data['pub_rec_bankruptcies'].fillna(0)
test_data['verification_status'].replace('Verified', 1, inplace=True)
# technically encoding two categories into one... might want to change
test_data['verification_status'].replace('Verified', 1, inplace=True)
test_data['verification_status'].replace('Source Verified', 1, inplace=True)
test_data['verification_status'].replace('Not Verified', 0, inplace=True)
test_data['initial_list_status'].replace('w', 1, inplace=True)
test_data['initial_list_status'].replace('f', 0, inplace=True)

train_data['mort_acc'] = train_data['mort_acc'].fillna(0)

test_data['mort_acc'] = test_data['mort_acc'].fillna(0)
# Strip percent(%) from int_rate
train_data['int_rate'] = train_data['int_rate'].str.rstrip('%').astype(float)
test_data['int_rate'] = test_data['int_rate'].str.rstrip('%').astype(float)

#Strip percent(%) from revol_util
train_data['revol_util'] = train_data['revol_util'].str.rstrip('%').astype(float)
test_data['revol_util'] = test_data['revol_util'].str.rstrip('%').astype(float)
train_data['revol_util'] = train_data['revol_util'].fillna(0)
test_data['revol_util'] = test_data['revol_util'].fillna(0)

X_train = train_data.iloc[:,:-1]
y_train = train_data.iloc[:,-1]
X_test = test_data.iloc[:,:]
X_train_numeric = X_train.select_dtypes(include=np.number)
X_test_numeric = X_test.select_dtypes(include=np.number)
# y_train_numeric = y_train.select_dtypes(include=np.number)
y_train_numeric = y_train.copy(deep=False)
y_train_numeric.replace('Fully Paid', 0.0, inplace=True)
y_train_numeric.replace('Charged Off', 1.0, inplace=True)


for col in X_train_numeric.columns:
    if X_train_numeric[col].isnull().values.any() > 0:
        print(col)
    x_min = X_train_numeric[col].min()
    x_max = X_train_numeric[col].max()
    X_train_numeric[col] = (X_train_numeric[col] - x_min) / (x_max - x_min)
    # also need to normalize test data, but must use training data
    X_test_numeric[col] = (X_test_numeric[col] - x_min) / (x_max - x_min)
    X_test_numeric[col].mask(X_test_numeric[col] < 0, 0, inplace=True)
    X_test_numeric[col].mask(X_test_numeric[col] > 1, 1, inplace=True)
    # TODO: DETERMINE WHAT TO DO IN THESE CASES!!!
    if X_test_numeric[col].min() < 0:
        print(f"min is less than zero in test data in column {col}")
        print(X_test_numeric[col].min())
    if X_test_numeric[col].max() > 1:
        print(f"max is more than one in test data in column {col}")
        print(X_test_numeric[col].max())


print("BEFORE data AUGMENTATION")
print(X_train_numeric.shape)
print(y_train_numeric.shape)

X_balanced = []
Y_balanced = []
for i in range(X_train_numeric.shape[0]):
    curr_df = X_train_numeric.iloc[i,:]
    d = curr_df.to_dict()
    label = y_train_numeric[i]
    if label == 1:
        for _ in range(3):
            X_balanced.append(d)
            Y_balanced.append(label)
    X_balanced.append(d)
    Y_balanced.append(label)
X_train_numeric = pd.DataFrame(X_balanced)
y_train_numeric= pd.Series(Y_balanced)

print('AFTER DATA AUGMENTATION')
print(X_train_numeric.shape)
print(y_train_numeric.shape)


# Custom Dataset
from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        """
        X: a dataframe with rows as training examples and columns features
        Y: a pandas series with labels"""
        self.x_train=torch.tensor(X.values, dtype=torch.float32)
        self.y_train=torch.tensor(Y.values, dtype=torch.float32)
        self.x_train = torch.reshape(self.x_train, (self.x_train.shape[0], 1, 1, self.x_train.shape[1],))

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return (self.x_train[idx], self.y_train[idx])


BATCH_SIZE = 64

n_epochs = 10 # paper used 100 originally
lr = 0.001

ds_train = CustomDataset(X_train_numeric, y_train_numeric)
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 3), stride=1, padding=0),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),

    # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=0),
    # nn.BatchNorm2d(64),
    # nn.ReLU(inplace=True),
    # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=0),
    # nn.BatchNorm2d(64),
    # nn.ReLU(inplace=True),

    nn.MaxPool2d(kernel_size=(1, 2), stride=1),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=0),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),

    # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=0),
    # nn.BatchNorm2d(64),
    # nn.ReLU(inplace=True),
    # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=0),
    # nn.BatchNorm2d(64),
    # nn.ReLU(inplace=True),
    # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=0),
    # nn.BatchNorm2d(64),
    # nn.ReLU(inplace=True),

    nn.MaxPool2d(kernel_size=(1, 2), stride=1),
    nn.Flatten(),
    nn.Linear(5056, 512),
    nn.ReLU(),
    nn.Linear(512, 1),
    nn.Sigmoid()
)
# our model has some # of parameters:
param_count = 0
for p in model.parameters():
    n_params = np.prod(list(p.data.shape)).item()
    param_count += n_params
print(f'total params: {param_count}')
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Define data loaders for train and val
train_loader = torch.utils.data.DataLoader(
                ds_train, 
                batch_size=BATCH_SIZE, shuffle=True)
for epoch in range(n_epochs):
    print(f'EPOCH {epoch}')
    model.train()
    train_loss = 0
    for i, data in enumerate(train_loader):
        features, labels = data
        # print(type(features))
        # print(f"PESKY BASTARDS: {features.isnan().sum()}")
        optimizer.zero_grad()
        # forward pass
        output = model(features)
        output = torch.reshape(output, (output.shape[0], ))
        if output.isnan().sum() > 0:
            print(f"RIP{output.isnan().sum()}")
        output = torch.nan_to_num(output)
        # calculate loss
        loss = criterion(output, labels)
        # backward pass
        loss.backward()
        # update
        optimizer.step()
        # track training loss
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"\ttraining loss: {train_loss}")


# Test Dataset

class TestDataset(Dataset):
    def __init__(self, X):
        """
        X: a dataframe with rows as training examples and columns features
        Y: a pandas series with labels"""
        self.x_train=torch.tensor(X.values, dtype=torch.float32)
        self.x_train = torch.reshape(self.x_train, (self.x_train.shape[0], 1, 1, self.x_train.shape[1],))

    def __len__(self):
        print(f'GETTING LENGTH {self.x_train.shape[0]}')
        return self.x_train.shape[0]

    def __getitem__(self,idx):
        return self.x_train[idx]


ds_test = TestDataset(X_test_numeric)

test_loader = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False)

output = None
# evaluate
with torch.no_grad():
    model.eval()
    val_loss = 0
    for i, features in enumerate(test_loader):
        output = model(features)
        output = torch.reshape(output, (output.shape[0], ))
        output = torch.nan_to_num(output)

predictions = output.cpu().detach().numpy()
assert len(predictions) == len(id_column)
submission_data = {"id": [], "loan_status": []}
for i in range(len(predictions)):
    submission_data['id'].append(id_column[i])
    submission_data['loan_status'].append(predictions[i])
predictions_df = pd.DataFrame(data=submission_data)
predictions_df.head()
predictions_df.to_csv("cnn_predictions_10_epochs", index=False)