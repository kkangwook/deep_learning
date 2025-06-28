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
-> CNN 필터32개->풀링2-> cnn필터 64 -> 풀링2로 모델 생성 -> 요약+시각화(keras.utils.plot_model(model, show_shapes=True))
-> 파라미터 개수 계산 -> 학습 밑에꺼 넣어서
  checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.keras',save_best_only=True)
  early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)
-> 손실그래프 그리기 ->베스트모델 가져오기-> 검증세트 평가 -> x_val[0]예측, y_val과 비교 -> 테스트 세트 검증
-> 가중치 시각화 하기: 첫번째 층으로 -> 가중치의 shape과 의미는? 
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


3. RNN
text가 RNN에 들어가기 위해서 거쳐야할과정은? ->같은 크기여야할 데이터는? -> (4,3,5)가 되었다면 각각의 의미는? 
-> 은닉차원이 6일때 입력가중치, 순환가중치, 절편개수는? -> 타임스텝크기라는 단어와 일치하는 단어 두가지는?
-> return_sequence=True/False각각일때 하나의 샘플당 은닉값은? -> 샘플까지 포함하면?

from tensorflow.keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=300) 
-> 이 데이터 구조확인하고 x_train[0]의 토큰수 확인하기 -> train세트에서 validation세트 20퍼떼기
-> 자를만한 위치 찾기 -> 우리는 100개까지, 잘렸을때 어떤식으로 되고 그 이유는?
-> RNN층  파라미터 5가지는? -> RNN층 2개 (3종류중 2개 섞어서)로 해서 모델생성(드롭아웃적용)
-> 요약보고 입력값크기는? 가중치개수는? RNN층 지난 은닉값 크기 두가지는?
----
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.keras',save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
----
-> 배치는 오히려 32개나 64개 넣어야 빠름(GPU의 병렬)+ simpleRNN보다 LSTM등이 시간단축 훨씬더 압도적으로-> 학습 -> 손실함수그래프보기 -> 제일 좋은 모델 불러오기
-> x_test와 y_test검증 -> x_test[:10] 예측도 해보기

spam-data로 이진분류해보기
bert나 복잡한 순환신경망은 불용어, 스템화등의 작업없이 그대로 Tokenizer로
임베딩할때는 input dim 더 넉넉하게 잡아도 됌
