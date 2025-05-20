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


2. CNN
-(x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()
-> 데이터 전처리 ->train세트에서 20% validation 세트 분리
-> 1D, 2D, 3D에 들어가기 위한 데이터 크기와 input_shape은?
-> CNN층의 파라미터 6개는? 가능한 풀링층 2개는?
-> CNN 필터32개->풀링2-> cnn필터 64 -> 풀링2로 모델 생성, 학습 밑에꺼 넣어서
  checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.keras',save_best_only=True)
  early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)
-> 손실그래프 그리기 ->베스트모델 가져오기-> 검증세트 평가 -> x_val[0]예측, y_val과 비교 -> 테스트 세트 검증
-> 가중치 시각화 하기 
fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):    # 16개 이미지 2줄로
        axs[i, j].imshow(weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)   #(3,3,1,1)에서 (3,3,1,32)까지
        axs[i, j].axis('off')                              #vmin과 vmax로 컬러맵으로 표현할 최소최대값 지정

plt.show() 
-> api모델 만들기: 합성곱층 첫번째까지 모델 만들기(두번째꺼까지 넣는법은?)->  특성맵 시각화
fig, axs = plt.subplots(4, 8, figsize=(15,8))   #n번째 층의 필터개수를 subplot의 가로세로개수로 (64면 8*8) 
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(fm_n_k[0,:,:,i*8 + j]) #하나 샘플의 (28,28)을 전부 가져오겠다
        axs[i, j].axis('off')
plt.show()

