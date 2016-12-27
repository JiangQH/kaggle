import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../dataset/train.csv')
test = pd.read_csv('../dataset/test.csv')
#print train.head()
#print train.tail()
print "-------------------------------"
#print test.head()
# drop useless data
train = train.drop(['Name', 'PassengerId', 'Ticket'], axis=1)
test = test.drop(['Name', 'Ticket'], axis=1)

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

# show the survived/not survived with sex
fig = plt.figure(figsize=(15, 6))







