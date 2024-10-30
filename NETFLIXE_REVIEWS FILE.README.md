 문제명
함수
코드내용
1.도전과제 달성 조건 : head, tail, 데이터셋 불러오기
pd.read_csv: 데이터셋 불러오기
coulmns: 컬럼 뭐있는지 확인
head: 상위 n개 행 보기
tail: 하위 n개 행 보기
netflix_df = pd.read_csv('netflix_reviews.csv')
print("\n컬럼확인함수:\n", netflix_df.columns)) 
print("\nhead함수:\n", netflix_df.head(10)) 
print("\ntail함수:\n",netflix_df.tail(8))
2.데이터 전처리: 불용어 제거
re.sub: 문자열 치환
apply: 함수 적용
import re
preprocess_text = lambda text: ("" if isinstance(text, float) else re.sub(r'\d+', '', re.sub(r'[^\w\s]', '', text.lower())).strip())
for column in netflix_df.select_dtypes(include=['object']).columns:
netflix_df[column] = netflix_df[column].apply(preprocess_text)
3.리뷰 카운트 시각화: 1~5점 데이터 분포 막대 그래프
value_counts: 값 세기
barplot: 시각화
리뷰 점수(1 ~ 5)의 분포를 시각화합니다.
import seaborn as sns
import matplotlib.pyplot as plt
review_counts = netflix_df['score'].value_counts().reset_index()
review_counts.columns = ['score', 'count']
palette = ['blue', 'orange', 'green', 'red', 'purple']
sns.barplot(x='score', y='count', hue='score', data=review_counts, palette=palette, legend=False)
plt.show()


#4번 문제 

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# 데이터 로드 및 전처리
netflix_df = pd.read_csv('netflix_reviews.csv')  # 넷플릭스 리뷰 데이터셋을 불러옵니다.
netflix_df['content'] = netflix_df['content'].fillna('')  # 'content' 열의 결측값을 빈 문자열로 채웁니다.
netflix_df['score'] = netflix_df['score'].fillna('')  # 'score' 열의 결측값을 빈 문자열로 채웁니다.
netflix_df = netflix_df[netflix_df['content'].str.len() > 1]  # 리뷰의 길이가 1보다 큰 데이터만 선택합니다.

# 토큰화 및 어휘집 구축
tokenizer = get_tokenizer('basic_english')  # 기본 영어 토크나이저를 사용합니다.
def yield_tokens(data_iter):  # 데이터 반복자에서 토큰을 생성하는 함수입니다.
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(netflix_df['content']), specials=["<unk>"])  # 어휘집을 생성합니다.
vocab.set_default_index(vocab["<unk>"])  # "<unk>" 토큰의 기본 인덱스를 설정합니다.

# 점수 인코딩
label_encoder = LabelEncoder()  # 레이블 인코더를 사용합니다.
netflix_df['score'] = label_encoder.fit_transform(netflix_df['score'])  # 점수를 인코딩합니다.

# 훈련-테스트 데이터 분할
train_df, test_df = train_test_split(netflix_df, test_size=0.2, random_state=42)  # 80%는 훈련 데이터, 20%는 테스트 데이터로 분할합니다.

# 커스텀 데이터셋
class ReviewDataset(Dataset):  # 데이터셋 클래스 정의
    def __init__(self, df):
        self.texts = df['content'].tolist()  # 텍스트 리스트를 가져옵니다.
        self.labels = df['score'].tolist()  # 레이블 리스트를 가져옵니다.

    def __len__(self):
        return len(self.texts)  # 데이터셋의 길이를 반환합니다.

    def __getitem__(self, idx):
        text = self.texts[idx]  # 인덱스에 따라 텍스트를 가져옵니다.
        label = self.labels[idx]  # 인덱스에 따라 레이블을 가져옵니다.
        return torch.tensor(vocab(tokenizer(text)), dtype=torch.long), torch.tensor(label, dtype=torch.float)  # 텍스트와 레이블을 반환합니다.

# 패딩을 위한 커스텀 콜레이트 함수
def collate_fn(batch):  # 배치 내 텍스트와 레이블을 패딩합니다.
    texts, labels = zip(*batch)  # 배치에서 텍스트와 레이블을 분리합니다.
    texts_padded = pad_sequence(texts, batch_first=True)  # 텍스트를 패딩합니다.
    labels_tensor = torch.stack(labels)  # 레이블을 텐서로 변환합니다.
    return texts_padded, labels_tensor  # 패딩된 텍스트와 레이블을 반환합니다.

# 데이터로더 생성
train_dataset = ReviewDataset(train_df)  # 훈련 데이터셋 생성
test_dataset = ReviewDataset(test_df)  # 테스트 데이터셋 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)  # 훈련 데이터로더 생성
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)  # 테스트 데이터로더 생성

# LSTM 모델 정의
class LSTMModel(nn.Module):  # LSTM 모델 클래스 정의
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 임베딩 층 정의
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # LSTM 층 정의
        self.fc = nn.Linear(hidden_dim, 1)  # 완전 연결 층 정의

    def forward(self, x):  # 순전파(forward) 함수 정의
        embedded = self.embedding(x)  # 임베딩 층 통과
        lstm_out, _ = self.lstm(embedded)  # LSTM 층 통과
        out = self.fc(lstm_out[:, -1, :])  # 마지막 출력값을 완전 연결 층에 통과
        return out  # 출력 반환

# 모델, 손실 함수, 최적화 기법 초기화
vocab_size = len(vocab)  # 어휘집 크기 설정
embedding_dim = 100  # 임베딩 차원 설정
hidden_dim = 128  # 히든 차원 설정
model = LSTMModel(vocab_size, embedding_dim, hidden_dim)  # 모델 초기화
criterion = nn.MSELoss()  # 평균 제곱 오차 손실 함수 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 최적화 기법 사용

# 훈련 루프
num_epochs =   # 에포크 수 설정
for epoch in range(num_epochs):
    model.train()  # 모델을 훈련 모드로 설정
    for texts, labels in train_loader:  # 훈련 데이터로 반복
        optimizer.zero_grad()  # 기울기 초기화
        outputs = model(texts)  # 모델 출력 계산
        loss = criterion(outputs.squeeze(), labels)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')  # 에포크와 손실 출력

# 11. 평가
model.eval()
with torch.no_grad():
    total_loss = 0
    for texts, labels in test_loader:
        outputs = model(texts)
        loss = criterion(outputs.squeeze(), labels)
        total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {avg_loss}')

# 새로운 리뷰에 대한 예측
new_review = "This app is great but has some bugs."  # 새로운 리뷰

# 리뷰를 토큰화하고 인덱스로 변환
tokenized_review = vocab(tokenizer(new_review))  # 토큰화 및 어휘집 인덱스 변환
input_tensor = torch.tensor(tokenized_review, dtype=torch.long).unsqueeze(0)  # 배치 차원 추가

# 예측 점수 계산
model.eval()  # 모델을 평가 모드로 설정
with torch.no_grad():  # 기울기 계산 비활성화
    predicted_score = model(input_tensor)  # 모델을 통해 예측
    predicted_score = predicted_score.item()  # 텐서를 스칼라 값으로 변환

print(f'Predicted Score: {predicted_score}')  # 예측 점수 출력
Epoch 1/5, Loss: 0.7553587555885315
Epoch 2/5, Loss: 0.7880595922470093
Epoch 3/5, Loss: 0.7075854539871216
Epoch 4/5, Loss: 0.6719469428062439
Epoch 5/5, Loss: 1.195376992225647
Average Test Loss: 0.9529675829320983
Predicted Score: 3.0822131633758545

겨

pip install tabulate # tabulate을 이용해서 표를 출력한 것이라 이것을 써야 합니다. 

 
 

#5번 문제 
import pandas as pd
from textblob import TextBlob

from IPython.display import display
# 데이터 불러오기
df = pd.read_csv('netflix_reviews.csv')

# 감성 분석 함수 정의
def get_sentiment(text):
    if isinstance(text, str):  # 문자열인 경우에만 분석
        return TextBlob(text).sentiment.polarity
    return 0  # 문자열이 아닐 경우 기본값 설정

# NaN 및 비문자열 처리
df['content'] = df['content'].fillna('')  # NaN을 빈 문자열로 대체

# 감성 분석 적용
df['sentiment'] = df['content'].apply(get_sentiment)

# 감성 라벨링
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))


# 원하는 열만 선택
result_df = df[['content', 'score', 'sentiment_label']].copy()  # copy() 메서드 호출
result_df.rename(columns={'content': 'content-c'}, inplace=True)
df.to_string(index=False) 
# Jupyter Notebook에서 HTML 형식으로 출력

from tabulate import tabulate


print(tabulate(result_df.head(5), headers='keys', tablefmt='grid', showindex=False))


5번 문제  각 부분 설명
_________________________________________________________________________________________________


1. 데이터 불러오기
df = pd.read_csv('netflix_reviews.csv')
설명: pandas를 사용하여 CSV 파일에서 리뷰 데이터를 읽어옵니다.
변수:
df: CSV 파일에서 읽어온 데이터프레임으로, 리뷰 데이터가 저장됩니다.




_________________________________________________________________________________________________


2. 감성 분석 함수 정의
def get_sentiment(text):
    if isinstance(text, str):  # 문자열인 경우에만 분석
        return TextBlob(text).sentiment.polarity
    return 0  # 문자열이 아닐 경우 기본값 설정
설명: TextBlob을 사용해 주어진 텍스트의 감성을 분석하는 함수입니다.
목적: 입력된 텍스트가 문자열인지 확인하여 안정성을 높이고, 감성 점수를 계산합니다.



_________________________________________________________________________________________________



3. NaN 및 비문자열 처리
df['content'] = df['content'].fillna('')  # NaN을 빈 문자열로 대체
설명: content 열의 NaN 값을 빈 문자열로 대체하여 이후 분석에서 발생할 수 있는 오류를 방지합니다.


_________________________________________________________________________________________________



4. 감성 분석 적용
df['sentiment'] = df['content'].apply(get_sentiment)
설명: apply 메소드를 사용하여 각 리뷰의 감성 점수를 계산하고 sentiment 열에 저장합니다.


_________________________________________________________________________________________________



5. 감성 라벨링
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))
설명: 감성 점수에 따라 'positive', 'negative', 'neutral'로 라벨링하여 sentiment_label 열에 저장합니다.



_________________________________________________________________________________________________




6. 원하는 열만 선택
result_df = df[['content', 'score', 'sentiment_label']].copy()  # copy() 메서드 호출
result_df.rename(columns={'content': 'content-c'}, inplace=True)
설명: content, score, sentiment_label 열만 선택하여 새로운 데이터프레임을 만듭니다. content 열의 이름을 content-c로 변경합니다.



_________________________________________________________________________________________________




7. 결과 출력
print(result_df.to_string(index=False))
설명: 결과 데이터프레임을 문자열 형태로 출력합니다. index=False는 인덱스 열을 출력하지 않도록 설정합니다.



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#6번 문제 풀기전에 설치해야 하는 모델 설치 명령어

pip install wordcloud

import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob  # TextBlob 임포트 추가


# 데이터 불러오기
df = pd.read_csv('netflix_reviews.csv')  # 파일 경로 확인

# 감성 분석을 통해 'negative' 리뷰를 분류해야 함

def get_sentiment(text):
    if isinstance(text, str):
        return TextBlob(text).sentiment.polarity
    return 0

df['sentiment'] = df['content'].apply(get_sentiment)
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))

# 부정적인 리뷰만 모으기
negative_reviews = " ".join(df[df['sentiment_label'] == 'negative']['content'])

# 불용어 설정
stopwords = set(STOPWORDS)
stopwords.update(['netflix', 'movie', 'show', 'time', 'app', 'series', 'phone'])

# 워드클라우드 생성
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(negative_reviews)

6번 문제 해설 



_________________________________________________________________________________________________



1. 데이터 불러오기
import pandas as pd
# 데이터 불러오기
df = pd.read_csv('netflix_reviews.csv')  # 파일 경로 확인
설명: pandas 라이브러리를 사용하여 CSV 파일에서 리뷰 데이터를 읽어옵니다.
변수:
df: CSV 파일에서 읽어온 데이터프레임으로, 리뷰 데이터가 저장됩니다.
목적: 데이터를 메모리에 로드하여 이후 분석에 사용할 수 있도록 준비합니다.


_________________________________________________________________________________________________


2. 감성 분석 함수 정의

from textblob import TextBlob  # TextBlob 임포트 추가
def get_sentiment(text):
    if isinstance(text, str):  # text가 문자열인지 확인
        return TextBlob(text).sentiment.polarity
    return 0  # 문자열이 아닐 경우 0 반환
설명: TextBlob을 사용해 주어진 텍스트의 감성을 분석하는 함수입니다.
함수:
get_sentiment(text): 주어진 텍스트가 문자열인 경우에만 감성 점수를 계산하여 -1(부정)에서 1(긍정) 사이의 값을 반환합니다. 문자열이 아닐 경우 0을 반환합니다.
isinstance의 역할:
isinstance(text, str): 이 조건문은 text가 문자열인지 확인합니다. 만약 text가 문자열이면 감성 점수를 계산하고, 그렇지 않으면 0을 반환합니다. 이 검사는 데이터가 예기치 않은 형식(예: None, 숫자 등)일 경우 발생할 수 있는 오류를 방지합니다.
목적: 리뷰 텍스트의 감성을 정량적으로 평가하기 위한 기반 함수를 정의하며, 입력 데이터의 형식을 확인하여 안정성을 높입니다.


_________________________________________________________________________________________________


3. 감성 분석 적용

# 감성 분석 적용
df['sentiment'] = df['content'].apply(get_sentiment)  # get_sentiment 함수 사용
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))
설명: apply 메소드를 사용해 각 리뷰의 감성 점수를 계산하고 라벨을 부여합니다.
apply 메소드의 역할:
df['content'].apply(get_sentiment): 각 리뷰 텍스트에 대해 get_sentiment 함수를 호출하여 감성 점수를 계산하고 sentiment 열에 저장합니다.
df['sentiment'].apply(lambda x: ...): 감성 점수에 따라 'positive', 'negative', 'neutral'로 라벨링하여 sentiment_label 열에 저장합니다.
목적: 리뷰의 감성을 정리하고, 후속 분석을 위한 부정적인 리뷰를 추출하기 위한 기초 데이터를 설정합니다.


_________________________________________________________________________________________________


4. 부정적인 리뷰 수집

# 부정적인 리뷰만 모으기
negative_reviews = " ".join(df[df['sentiment_label'] == 'negative']['content'])
설명: 부정적인 리뷰를 필터링하여 하나의 문자열로 결합합니다.
변수:
negative_reviews: 부정적인 리뷰 내용을 하나의 문자열로 결합한 것.
목적: 후속 분석인 워드클라우드 생성을 위한 데이터를 준비합니다.



_________________________________________________________________________________________________


5. 불용어 설정

from wordcloud import WordCloud, STOPWORDS

# 불용어 설정
stopwords = set(STOPWORDS)
stopwords.update(['netflix', 'movie', 'show', 'time', 'app', 'series', 'phone'])
설명: 워드클라우드 생성 시 사용할 불용어 목록을 설정합니다.
변수:
stopwords: 분석에서 제외할 단어들의 집합으로, 일반적으로 자주 사용되는 단어들입니다.
목적: 유의미한 단어만 시각화하여 데이터의 의미를 잘 전달하기 위해 불필요한 단어를 제거합니다.



_____________________________________________________________________________________________________________________________________________



6. 워드클라우드 생성

# 워드클라우드 생성
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(negative_reviews)

# 워드클라우드 시각화 (선택적으로 추가)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # 축 제거
plt.show()
설명: 수집한 부정적인 리뷰를 바탕으로 워드클라우드를 생성하고 시각화합니다.
변수:
wordcloud: 생성된 워드클라우드 객체로, 부정적인 리뷰에서 자주 나타나는 단어를 시각적으로 표현합니다.
목적: 부정적인 리뷰의 패턴을 이해하고, 어떤 단어가 빈번하게 사용되는지를 시각화하여 분석합니다.

