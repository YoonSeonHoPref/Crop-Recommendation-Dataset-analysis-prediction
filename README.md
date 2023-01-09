# Crop-Recommendation-Dataset-analysis-prediction
농작물 데이터 분석 및 예측 (개인 프로젝트) <br>
https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset
<hr>

df=pd.read_csv("crop.csv")<br>
df<br>
![image](https://user-images.githubusercontent.com/111934213/211227584-2135af44-d0fc-45d3-8209-35421df1a43d.png)<br>

df.info <br>
![image](https://user-images.githubusercontent.com/111934213/211227595-ceac37a7-645b-4ea5-a735-d6581c5bfdbf.png)<br>

print("label 개수 : ",len(df['label'].value_counts())) <br>
#### label 개수 : 22 <br>

df['label'].value_counts() <br>
![image](https://user-images.githubusercontent.com/111934213/211227812-3cc4283a-0911-45db-836b-374e0e43bbf8.png) <br>


- columns : 8개 [N, P,	K,	temperature,	humidity,	ph,	rainfall,	label] <br>
- 2200개 데이터 , label 별 100개 
