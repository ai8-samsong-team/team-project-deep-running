타이타닉 데이터셋 
넷플릭스 리뷰파일  문제명
함수
코드내용
1.데이터셋 불러오기 
load_dataset

코드
import seaborn as sns
titanic = sns.load_dataset('titanic')

seaborn를 이용하여 데이셋 불러오는 문제였고

2.데이터셋 feature 분석 
2-1
head
2-2
describe

2-3
count(수를 센다), mean(평균), min(최소값), max(최대값), 25%(1사분위수(Q1)), 50% (2사분위수) , 75% (3사분위수), std(평균편차)
2-4
isnull
sum
코드
titanic.head()
titanic.describe()
2-3 count(수를 센다), mean(평균), min(최소값), max(최대값), 25%(1사분위수(Q1)), 50% (2사분위수) , 75% (3사분위수), std(평균편차)
titanic.isnull().sum()



3.데이터셋 정제
isnull
fillna
map
print(titanic.isnull().sum()) 
titanic['age'] = titanic['age'].fillna(titanic['age'].median()) 
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0]) 
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1}) 
titanic['alive'] = titanic['alive'].map({'no': 1, 'yes': 0}) 
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2}) 

map 으로 파일값을 지정하여 바꾸고 그리고 none 값을 ""로 바꾸어 결즉치를 없앴다.

