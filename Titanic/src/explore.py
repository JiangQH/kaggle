import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv('../dataset/train.csv')
test = pd.read_csv('../dataset/test.csv')
#print train.head()
#print train.tail()
print "-------------------------------"
#print test.head()
# drop useless data
train = train.drop(['Name', 'PassengerId', 'Ticket'], axis=1)
test = test.drop(['Name', 'Ticket'], axis=1)

'''
# Data Exploration
# explore some features
plt.rc('font', size=13)
fig = plt.figure(figsize=(18, 8))
alpha = 0.6

# the age distribution
ax1 = plt.subplot2grid((2, 3), (0, 0))
train.Age.value_counts().plot(kind='kde', color='#FA2379', label='train', alpha=alpha)
test.Age.value_counts().plot(kind='kde', label='test', alpha=alpha)
ax1.set_xlabel('Age')
ax1.set_title('age distribution')
plt.legend(loc='best')

# the Pclass distribution
ax2 = plt.subplot2grid((2, 3), (0, 1))
train.Pclass.value_counts().plot(kind='barh', color='#FA2379', label='train', alpha=alpha)
test.Pclass.value_counts().plot(kind='barh', label='test', alpha=alpha)
ax2.set_xlabel('frequency')
ax2.set_ylabel('Pclass')
plt.legend(loc='best')

# the gender
ax3 = plt.subplot2grid((2, 3), (0, 2))
train.Sex.value_counts().plot(kind='barh', color='#FA2379', label='train', alpha=alpha)
test.Sex.value_counts().plot(kind='barh', label='test', alpha=alpha)
ax3.set_xlabel('frequency')
ax3.set_ylabel('Sex')
plt.legend(loc='best')

# the Passenger Fare
ax4 = plt.subplot2grid((2, 3), (1, 0))
train.Fare.value_counts().plot(kind='kde', color='#FA2379', label='train', alpha=alpha)
test.Fare.value_counts().plot(kind='kde', label='test', alpha=alpha)
ax4.set_xlabel('Fare')
ax4.set_title('fare distribution')
plt.legend(loc='best')

# the Embarked
ax5 = plt.subplot2grid((2, 3), (1, 1))
train.Embarked.value_counts().plot(kind='barh', color='#FA2379', label='train', alpha=alpha)
test.Embarked.value_counts().plot(kind='barh', label='test', alpha=alpha)
ax5.set_xlabel('frequency')
ax5.set_ylabel('Embarked')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

print train.Survived.value_counts()


# show survived/not survived distribution with age
fig = plt.figure(figsize=(12, 5))
train[train.Survived == 0].Age.value_counts().plot(kind='density', color='#FA2379',
                                                   label='Not survived', alpha=alpha)
train[train.Survived == 1].Age.value_counts().plot(kind='density', label='Survived', alpha=alpha)
plt.xlabel('Age')
plt.title('Age distribution')
plt.legend(loc='best')
plt.grid()
plt.show()

# show the survived/not survived with sex(male/femal)
df_male = train[train.Sex=='male'].Survived.value_counts().sort_index()
df_female = train[train.Sex=='female'].Survived.value_counts().sort_index()
fg = plt.figure(figsize=(18, 6))

ax1 = plt.subplot2grid((1, 2), (0, 0))
df_female.plot(kind='barh', color='#FA2379', label='female', alpha=alpha)
df_male.plot(kind='barh', label='male', alpha=alpha)
ax1.set_xlabel('Frequency')
ax1.set_ylabel(['died', 'survived'])
ax1.set_title("Who will survived with respect to sex?" )
plt.legend(loc='best')
plt.grid()

ax2 = plt.subplot2grid((1,2), (0,1))
(df_female/train[train.Sex=='female'].shape[0]).plot(kind='barh', color='#FA2379', label='Female', alpha=alpha)
(df_male/train[train.Sex=='male'].shape[0]).plot(kind='barh', label='Male', alpha=alpha-0.1)
ax2.set_xlabel('Rate')
ax2.set_yticklabels(['Died', 'Survived'])
ax2.set_title("What's the survived rate with respect to sex?" )
plt.legend(loc='best')
plt.grid()


# the class that influence the survived
df_male = train[train.Sex=='male']
df_female = train[train.Sex=='female']
fig = plt.figure(figsize=(18, 6))

ax1 = plt.subplot2grid((1,4), (0,0))
df_female[df_female.Pclass<3].Survived.value_counts().sort_index().plot(kind='bar', color='#FA2379', alpha=alpha)
ax1.set_ylabel('Frequrncy')
ax1.set_ylim((0,350))
ax1.set_xticklabels(['Died', 'Survived'])
ax1.set_title("How will high-class female survived?", y=1.05)
plt.grid()

ax2 = plt.subplot2grid((1,4), (0,1))
df_female[df_female.Pclass==3].Survived.value_counts().sort_index().plot(kind='bar', color='#23FA79', alpha=alpha)
ax2.set_ylabel('Frequrncy')
ax2.set_ylim((0,350))
ax2.set_xticklabels(['Died', 'Survived'])
ax2.set_title("How will low-class female survived?", y=1.05)
plt.grid()

ax3 = plt.subplot2grid((1,4), (0,2))
df_male[df_male.Pclass<3].Survived.value_counts().sort_index().plot(kind='bar', color='#00FA23', alpha=alpha)
ax3.set_ylabel('Frequrncy')
ax3.set_ylim((0,350))
ax3.set_xticklabels(['Died', 'Survived'])
ax3.set_title("How will high-class male survived?", y=1.05)
plt.grid()

ax4 = plt.subplot2grid((1,4), (0,3))
df_male[df_male.Pclass==3].Survived.value_counts().sort_index().plot(kind='bar', color='#2379FA', alpha=alpha)
ax4.set_ylabel('Frequrncy')
ax4.set_ylim((0,350))
ax4.set_xticklabels(['Died', 'Survived'])
ax4.set_title("How will low-class male survived?", y=1.05)
plt.grid()
plt.tight_layout()
plt.show()
'''

print train.isnull().sum()
print test.isnull().sum()

# data cleaning & feature engienering

# the embarked data
print train['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].fillna("S")
embarked_dummies_train = pd.get_dummies(train['Embarked'])
embarked_dummies_test = pd.get_dummies(test['Embarked'])
# append it to the data and drop the original column
train = train.join(embarked_dummies_train)
test = test.join(embarked_dummies_test)
train.drop(['Embarked'], axis=1, inplace=True)
test.drop(['Embarked'], axis=1, inplace=True)

# the cabin data
# print train['Cabin'].value_counts()
# maybe just drop it ? because it has too many nan
train.drop(['Cabin'], axis=1, inplace=True)
test.drop(['Cabin'], axis=1, inplace=True)

# the Fare data. fill the missing value for fare in test
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
# convert float to int
train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)

# the parch and sibsp can represent whether a person has a family on board. we use family to represent
train['Family'] = train['Parch'] + train['SibSp']
test['Family'] = test['Parch'] + test['SibSp']
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0
#drop the original
train.drop(['Parch'], axis=1, inplace=True)
test.drop(['Parch'], axis=1, inplace=True)


# the age data
fig = plt.figure(figsize=(15, 4))
alpha = 0.6
ax1 = plt.subplot2grid((2, 3), (0, 0))
train.Age.value_counts().plot(kind='kde', color='#FA2379', label='train', alpha=alpha)
test.Age.value_counts().plot(kind='kde', label='test', alpha=alpha)
ax1.set_xlabel('Age')
ax1.set_title('age distribution')
plt.legend(loc='best')
plt.show()
# get the mean / std for train and test
mean_train = train['Age'].mean()
val_train = train['Age'].std()
null_count_train = train['Age'].isnull().sum()
mean_test = test['Age'].mean()
val_test = test['Age'].std()
null_count_test = test['Age'].isnull().sum()
# generate random numbers for the age between (mean-std) & (mean + std)
rand_train = np.random.randint((mean_train-val_train), (mean_train+val_train), null_count_train)
rand_test = np.random.randint((mean_test- val_test), (mean_test + val_test), null_count_test)
# fill it
train['Age'][train['Age'].isnull()] = rand_train
test['Age'][test['Age'].isnull()] = rand_test











