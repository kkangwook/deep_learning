1. 기본 DNN
-(x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data() 불러오기
- 10개 이미지화 -> 데이터 전처리 ->train세트에서 20% validation 세트 분리
- 입력층에 들어가는 두가지 방법은?
- 층 2개, 규제층 하나로 -> 요약정보보기 -> 각 파라미터 개수 나온 원리는?+실제로 계산해보기
-밑에 애들 넣고 학습시키기
  checkpoint_cb=keras.callbacks.ModelCheckpoint('best-model.h5', save_best_only=True)  
  early_stopping_cb=keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)
- 손실값 이미지 비교, 어느 epoch에서 멈췄는지 보기 
- 최고 모델 불러와 검증세트 평가-> x_test[:5]예측하고 정답과 비교 -> 테스트 세트 평가

- data=pd.read_csv('../machine_learning/perch_3v.csv')얘로 회귀해보기
- fish=pd.read_csv('../machine_learning/fish.csv') 예로 분류해보기
