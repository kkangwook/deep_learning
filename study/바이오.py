transformer, GNN관련 논문
Deep Learning in Bioinformatics — Briefings in Bioinformatics (2021)
Transformers in Computational Biology — Nature Methods (2022)
Graph Neural Networks in Biology and Medicine — Nature Reviews Genetics (2022)

원서책
Deep Learning in Bioinformatics (Springer, 2022)


#유전자, 아미노산 서열을 통한 클래스 분류:
-CNN/RNN:	One-hot or 임베딩 행렬
  유전자는 (샘플 수, 시퀀스 길이, 4)의 크기로 들어감
  아미노산은 (샘플 수, 시퀀스 길이, 20)의 크기로 들어감
-Transformer:	토큰 인덱스 시퀀스 + 임베딩
  tensorflow의 Tokenizer같은거 사용: Tokenizer(char_level=True)로 해서 각 염기별로 토큰화하거나 k-mer로 나눠서 토큰화 
  -> 이후 패딩 -> word2vec이나 임베딩층으로 임베딩 
-그래프 신경망 (GNN): 단백질 3D 구조를 **그래프 노드(아미노산)**와 **엣지(결합, 거리)**로 표현


#합성경로: 분류(이 경로의 기능은? 어떤 종의 경로인지, 이 경로가 특정 화합물을 만들수있는지) or 회귀(경로의 반응속도나 효율은? )
- RNN, Transformer 등(순서가 중요): 효소(EC번호나) 유전자 종류를 토큰인덱싱 → 패딩 → 임베딩 → 시퀀스 모델
- 그래프 형태로 처리 (GNN 등): 경로를 그래프로 보고 노드(화합물/효소), 엣지(반응)를 입력 


프로젝트
예를 들어서 어떤 과의 식물에는 2000종이 있는데 어느 한 종에서만 고유의 물질을 분비해.
근데 이 물질의 합성경로는 밝혀지지 않았어. 나는 같은 과의 식물들에 속하는 다양한 종의 유전자 종류를 넣거나 효소번호를 넣어서
학습시키고 저 고유의 물질을 분비하는 종에서만 존재하는 유전자나 효소번호를 통해 합성경로를 예측하고 싶은데 가능할까? 
그리고 이건 분류야 회귀야?

유전자or효소를 벡터화(one-hot-encoding or tfidf) 
- 이진 분류일경우
RandomForest, XGBoost로 feature_importance를 통해 어느 유전자나 효소가 중요한지 예측가능
순서가 중요하면 Transformer, LSTM사용 
"이런 유전자/효소들이 핵심이다"라는 정보를 얻으면 -> KEGG/MetaCyc 데이터베이스에서 해당 유전자/효소가 참여하는 경로 검색
-> 가능한 합성 경로 조합을 재구성 (그래프 탐색 활용) -> 실험 검증 대상으로 narrowing 

- 클래스1의 샘플이 딱 하나일경우 
그 종에만 존재하는 유전자나 효소 찾고 
Transformer / Attention 계열 모델은 특정 유전자/효소에 더 집중했는지 시각화 가능
attention_weights를 분석하면, A 종에서 특이한 유전자가 높은 중요도를 가졌는지 확인 가능
