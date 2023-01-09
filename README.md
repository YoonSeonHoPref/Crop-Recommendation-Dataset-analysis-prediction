# Crop-Recommendation-Dataset-analysis-prediction


농작물 데이터 분석 및 예측 (개인 프로젝트 공부) 
https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset


## ※ 파일 로드 및 탐색 

import pandas as pd <br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
import seaborn as sns<br>

df=pd.read_csv("crop.csv")<br>
df<br>

![image](https://user-images.githubusercontent.com/111934213/211227584-2135af44-d0fc-45d3-8209-35421df1a43d.png)<br>

- N - ratio of Nitrogen content in soil (토양 내 질소의 양)
- P - ratio of Phosphorous content in soil (토양 내 인산의 양)
- K - ratio of Potassium content in soil (토양 내 칼륨의 양)
- temperature - temperature in degree Celsius (섭씨 온도)
- humidity - relative humidity in % (습도 %)
- ph - ph value of the soil (토지 ph)
- rainfall - rainfall in mm (강우량 mm)

df.info <br>

![image](https://user-images.githubusercontent.com/111934213/211227595-ceac37a7-645b-4ea5-a735-d6581c5bfdbf.png)<br>

print("label 개수 : ",len(df['label'].value_counts())) <br>
####  label 개수 : 22 <br>

df['label'].unique() <br>
![image](https://user-images.githubusercontent.com/111934213/211248377-d34acb6c-0c44-44ca-90ed-39e69acb6dad.png)


df['label'].value_counts() <br>

![image](https://user-images.githubusercontent.com/111934213/211227812-3cc4283a-0911-45db-836b-374e0e43bbf8.png) <br>


- columns : 8개 [N, P,	K,	temperature,	humidity,	ph,	rainfall,	label] <br>
- 총 2200개 데이터 ,22개의 label ,  label 별 100개씩 있음 , null 데이터는 x <br>
- 라벨 이름 : 'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
       'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
       'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
       'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'<br>

## 본인이 서치한 데이터 관련 지식 <br>

- 화학비료의 주성분은 크게 질소(N), 인산(P),칼륨(K) 등의 무기질 물질임
- 작물의 생장,번식을 위해 꼭 필요한 양분은 16종으로 작물필수원소라고 하는데, 필수원소 가운데 작물에 많이 필요한 질소, 인산, 칼륨은 일반 농지에선 부족하기 쉬움
- 위 데이터 간의 N,P,K의 양에 따라 농작물의 생장,번식의 차이가 있음
- 토양 PH는 토양의 산성 또는 알칼리성의 측정이고, 0(산성) ~ 15(알칼리성)의 범위임
- 많은 식물들과 농작물들의 형성은 알칼리성 또는 산성 조건을 선호함
- ph는 토양 화학물질 사이의 복잡한 상호작용에 영향을 줄 수 있음
- 예를 들어, 인산은 6.0과 7.5의 PH를 필요로 하며, 이 범위를 벗어나면 화학적으로 부동화됨
- 비는 토양 습도와 토양 PH에 영향을 줌
- 쌀 같은 경우 200mm 이상의 강우량과 80% 이상의 습도를 필요로 함
- 강우량과 습도, 비의 상관관계를 봐도 됨



## ※ 상관관계 

x= data[['N','P','K','temperature','humidity','ph','rainfall']] <br>
cor = x.corr() <br>

fig, ax = plt.subplots( figsize=(5,5) ) <br>


mask = np.zeros_like(cor, dtype=np.bool) <br>
mask[np.triu_indices_from(mask)] = True <br>
plt.title("Correlation of Features", y = 1, size = 15)<br>

sns.heatmap(cor, 
            cmap = 'RdYlBu_r', 
            annot = True,   
            mask=mask,      
            linewidths=.5,  
            cbar_kws={"shrink": .5},
            vmin = -1,vmax = 1)  <br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/111934213/211250367-00f84e08-17c6-445f-bf50-19bc37cd46c6.png) <br>

- 
           
