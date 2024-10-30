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


4-1 모델 학습 준비
titanic = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size']]
X = titanic.drop('survived', axis=1) # feature
y = titanic['survived'] # target

4-1 출력
X
pclass	sex	age	sibsp	parch	fare	embarked	family_size
0	3	0	22.0	1	0	7.2500	2	2
1	1	1	38.0	1	0	71.2833	0	2
2	3	1	26.0	0	0	7.9250	2	1
3	1	1	35.0	1	0	53.1000	2	2
4	3	0	35.0	0	0	8.0500	2	1
...	...	...	...	...	...	...	...	...
886	2	0	27.0	0	0	13.0000	2	1
887	1	1	19.0	0	0	30.0000	2	1
888	3	1	28.0	1	2	23.4500	2	4
889	1	0	26.0	0	0	30.0000	0	1
890	3	0	32.0	0	0	7.7500	1	1
891 rows × 8 columns

4-1 출력
y
0      0
1      1
2      1
3      1
4      0
..
886    0
887    1
888    0
889    1
890    0
Name: survived, Length: 891, dtype: int64

4-2 Logistic Regression
라이브러리 입력
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

예측
y_pred = model.predict(X_test)

평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
Accuracy: 0.8044692737430168
Classification Report:
precision    recall  f1-score   support


       0       0.82      0.86      0.84       105
       1       0.78      0.73      0.76        74

accuracy                           0.80       179
macro avg       0.80      0.79      0.80       179
weighted avg       0.80      0.80      0.80       179

4-3 Random Forest
from sklearn.tree import DecisionTreeClassifier

데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

모델 생성 및 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

예측
y_pred = model.predict(X_test)

평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
Accuracy: 0.770949720670391
Classification Report:
precision    recall  f1-score   support


       0       0.83      0.76      0.80       105
       1       0.70      0.78      0.74        74

accuracy                           0.77       179
macro avg       0.77      0.77      0.77       179
weighted avg       0.78      0.77      0.77       179

4-4 XGBoost
import xgboost as xgb
from sklearn.metrics import mean_squared_error

데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

XGBoost 모델 생성
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

모델 학습
xgb_model.fit(X_train_scaled, y_train)

예측
y_pred_xgb = xgb_model.predict(X_test_scaled)

평가
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost 모델의 MSE: {mse_xgb}')
XGBoost 모델의 MSE: 0.12981004899201257 머신러닝을 다양한모델을 이용한거야 각모델별로 단락을 나누고 각쓰인함수 그리고 그함수가 머신러닝에서 어떤역할을 해서 결국 최종목적인 머신러닝이 되는지 설명이 필요해 일반인이 보고 해설을 보고 설계가 이렇게 되서 이렇게 되는 거다 수준으로

답변


빠른 검색

아래는 다양한 머신러닝 모델을 사용한 분석 과정을 설명한 것입니다. 각 모델별로 사용된 함수와 그 역할을 간단히 설명하겠습니다.

4-1 모델 학습 준비
데이터 준비
먼저, 타이타닉 데이터셋에서 필요한 특성(X)과 목표 변수(y)를 준비합니다. 목표 변수는 생존 여부(survived)이며, 특성은 승객의 클래스(pclass), 성별(sex), 나이(age), 형제/자매 수(sibsp), 부모/자녀 수(parch), 요금(fare), 탑승지(embarked), 가족 크기(family_size)로 구성됩니다.


X = titanic.drop('survived', axis=1) # 특성
y = titanic['survived'] # 목표 변수
4-2 로지스틱 회귀 (Logistic Regression)
라이브러리 입력
로지스틱 회귀 모형을 사용하기 위해 필요한 라이브러리를 import합니다.




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
데이터 분할
데이터를 훈련 세트와 테스트 세트로 나눕니다. test_size=0.2는 전체 데이터의 20%를 테스트 세트로 사용하겠다는 의미입니다.




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
데이터 스케일링
특성의 스케일을 조정하여 모델의 성능을 향상시킵니다. StandardScaler는 평균이 0이고 표준편차가 1이 되도록 데이터를 변환합니다.



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
모델 생성 및 학습
로지스틱 회귀 모델을 생성하고 훈련 데이터를 사용하여 학습합니다.



model = LogisticRegression()
model.fit(X_train, y_train)
예측 및 평가
테스트 세트에 대해 예측한 후, 정확도와 분류 보고서를 출력합니다.



y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
로지스틱 회귀는 이진 분류 문제를 해결하는 데 사용되는 기법으로, 확률을 기반으로 한 예측을 제공합니다.

4-3 랜덤 포레스트 (Random Forest)
라이브러리 입력
랜덤 포레스트 모델을 사용하기 위해 DecisionTreeClassifier를 import합니다.




from sklearn.tree import DecisionTreeClassifier
데이터 분할 및 스케일링
로지스틱 회귀와 동일한 방식으로 데이터를 분할하고 스케일링합니다.

모델 생성 및 학습
랜덤 포레스트 대신 단일 결정 트리를 사용하여 학습합니다.




model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
예측 및 평가
테스트 세트에 대해 예측한 후, 정확도와 분류 보고서를 출력합니다.




y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
랜덤 포레스트는 여러 개의 결정 트리를 결합하여 예측 정확도를 높이며, 과적합을 방지하는 데 효과적입니다.

4-4 XGBoost
라이브러리 입력
XGBoost 모델을 사용하기 위해 필요한 라이브러리를 import합니다.




import xgboost as xgb
from sklearn.metrics import mean_squared_error
데이터 분할 및 스케일링
로지스틱 회귀와 동일한 방식으로 데이터를 분할하고 스케일링합니다.

XGBoost 모델 생성
XGBoost 회귀 모델을 생성하고 학습합니다.

python


xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
예측 및 평가
테스트 세트에 대해 예측한 후, 평균 제곱 오차(MSE)를 출력합니다.



y_pred_xgb = xgb_model.predict(X_test_scaled)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost 모델의 MSE: {mse_xgb}')
XGBoost는 Gradient Boosting 알고리즘을 기반으로 하며, 성능이 뛰어나고 대규모 데이터셋에 적합합니다.

결론
이러한 다양한 모델을 사용하여 타이타닉 데이터셋의 생존 여부를 예측하는 과정을 살펴보았습니다. 각 모델은 데이터 전처리, 학습, 예측 및 평가 단계에서 중요한 역할을 하며, 각각의 특성과 장점을 활용함으로써 최종적인 예측 성능을 향상시킬 수 있습니다.
목표:

타이타닉 호에서 승객이 생존했는지 여부를 예측하는 것입니다. 이는 이진 분류 문제로, 결과는 0(사망) 또는 1(생존)입니다.
출력 내용:

정확도 (Accuracy): 모델이 테스트 데이터에서 얼마나 정확하게 예측했는지를 나타내는 지표입니다. 예를 들어, "정확도: 80%"라는 결과는 모델이 80%의 경우를 올바르게 예측했다는 의미입니다.
분류 보고서 (Classification Report):
정밀도 (Precision): 모델이 생존자로 예측한 사람 중 실제로 생존한 비율입니다.
재현율 (Recall): 실제 생존자 중 모델이 올바르게 예측한 비율입니다.
F1-점수 (F1-Score): 정밀도와 재현율의 조화 평균으로, 모델의 전반적인 성능을 나타냅니다.


넷플릭스 리뷰 파일
