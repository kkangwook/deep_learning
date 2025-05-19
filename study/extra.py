딥러닝: 층 마다 선형방정식을 통과시켜 결과를 출력
  딥러닝의 선형방정식: 가중치와 절편을 랜덤하게 초기화 한 다음 에포크를 반복하면서 경사하강법으로 최적의 가중치와 절편을 찾음 

뉴련의 개수: 선형방정식의 개수
->은닉층의 뉴런개수가 100개면 100개의 선형방정식을 준비해 입력층의 모든 값을 입력층의모든값에 대응되는 가중치를 가진 선형방정식에 대입해 100개의 z값 출력-> 시그모이드나 렐루로 z값 변형
->출력층의 뉴런개수를 10개로 하면 10개의 선형방정식을 준비해 100개의 모든 z값을 하나 하나의 선형방정식에(100개의 가중치를 가짐) 넣어 10개의 z값으로 출력
-> 이후 소프트맥스 활성화 함수를 이용해 하나의 예측값으로 출력 

소프트맥스 함수: 모든 z값을 모아 하나의 결과를 출력(사실 하나의 결과는 아니고 각 클래스==z개수 별 확률 출력/이 확률의 전체합이 1)  
렐루함수: 각 z별 하나의 z에 렐루값이 적용돼 z개수만큼 출력 EX)z값이 10개면 렐루값도 10개 (z가 0보다 작으면 0, 0보다 크면 z로 변환)
시그모이드 함수: 얘도 렐루처럼 작용(2개가 들어와야한다고 착각할 수 있지만 사실 이진 분류에서도 시그모이드 함수는 하나에만 적용됐고 나머지는 1-a를 했었음) -> 0~1의 값으로 변환


!!!!!!!!!!!!!!!!딥러닝에서 가중치를 업데이트하는 방법!!!!!!!!!!!!!!!!!
1. 순전파 (Forward Pass)
입력 → 은닉층들 → 출력층 → softmax → 예측값 계산

2. 손실 함수 계산
예측값 vs 정답 → 손실(loss) 계산   예: categorical crossentropy

3. 역전파 (Backward Pass)
출력층부터 시작해서 오차(손실)를 역으로 전달하며 각 층의 가중치에 대해 미분(기울기)**을 계산

4. 가중치 업데이트
계산된 기울기(gradient)를 가지고 ***모든 층의*** 가중치를 동시에 업데이트 (ex: SGD, Adam 같은 옵티마이저로)



-----------------딥러닝에서의 선형대수학--------------------------
DNN
28*28을 1차원화 한다면 (768,)임(세로벡터)-> 그런데 tensorflow는 (배치크기,입력차원)으로 표시하므로 샘플하나를 (1,768)로 표현됌(가로)
그리고 뉴런개수를 10개라 한다면 가중치는 (768,10)이 됨-> 따라서 행렬곱하면 (1,768)@(768,10)=(1,10)의 구조로 나오개 됨(행렬곱을 가중치가 뒤에)
더 나아가 배치크기 32라면 (32,768)@(768,10)=(32,10) 이고 여기에 절편을 더하게 된다면 절편개수는 뉴런개수이므로 (10,)가되고
절편이 더해진다면 브로드캐스팅으로 32개 행마다 (1,10)으로 변환되어 더해짐-> 최종(32,10)

RNN
32개의 배치, 각 샘플은 다 다른 시퀀스길이를 가지고 있으며 가장 긴 길이를 10이라하때 패딩으로 전부 길이를 10, 각 토큰은 입력특성을 4개로->최종 입력크기는(32,10,4)
뉴런개수를 3이라하면 입력가중치는 (각 토큰별 입력특성개수,뉴런개수)=(4,3)-> 입력가중치를 곱하면 (32,10,4)@(4,3)은 (32,10,3)
순환가중치는 크기가 (뉴런개수,뉴런개수)인 (3,3)-> 순환가중치 곱하면 (32,10,3)@(3,3)->(32,10,3)
여기에 마지막으로 절편을 더하면 (뉴런개수,)인 (3,)가 브로드캐스팅으로 인해 (1,3)으로 전환돼 배열/행의 각 3열에 더해져 (32,10,3)
!!!!실제로는 (32,10,4)가 한꺼번에 들어가는게 아니라 (32,1,4)가 순차적으로 10번 들어감!!!!!!!!!
return_sequence=True면 모든 은닉값이 출력되느로 각 배치별 (10,3)이고 최종 결과는 (32,10,3)
return_sequence=False면 마지막 하나의 은닉값인 (3,)이고 최종 결과는 (1,32,3)



______
__________
_____________
사용법


-이진분류: 손실함수로 binary_crossentropy사용: 값이 클수록 손실큼
  클래스 1에 대한 손실= -log(h)
  클래스 0에 대한 손실= -log(1-h)

-다중분류: 그냥 -log(h) # h는 정답값의 확률
  정답이 [0,0,1,0]과 같은 원핫인코딩인경우 손실함수는 categorical_crossentropy 
  정답이 정수인 경우 손실함수로 sparse_categorical_crossentropy
  정답이 범주형 문자면 레이블 인코딩이나 원핫인코딩필요
정답이 [0, 1, 0]이고 소프트맥스통과z값이 [0.2, 0.7, 0.1]이면 손실= -log(0.7) = 0.3567
정답이 [0, 0, 1]인데 [0.2, 0.7, 0.1]로 나오면 손실= -log(0.1) = 2.3026


1. DNN(밀집층)
from tensorflow import keras
데이터 준비: (x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data() -> 10개 이미지화
->0~255값 사이면 255로 나눠 0~1로 정규화 -> x_train, y_train에서 validation세트 20%로 분리 
->데이터 들어갈때 보통 (100샘플수, 28*28)의 2차원 배열로 들어감
model=keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28))) #원래 1차원이면 필요X 
model.add(keras.layers.Dense(100,activation='relu',name='hidden')) #flatten없으면 input_shape=(n*n,), relu나 sigmoid
model.add(keras.layers.Dropout(0.2~0.5))  #규제할 층 바로 다음에 add, 파라미터는 내가 정함 
model.add(keras.layers.Dense(10,activation='softmax',name='output')) #마지막 층은 반드시 sigmoid or softmax or linear(회귀모델)
model.summary() #로 구조, 정보 확 
model.compile(optimizer='adam',            #여기 옵티마이저에 위의 'adam값 넣어 설정
              loss='sparse_categorical_crossentropy',metrics=['accuracy'])
               # binary or categorical or sparse_categorical (crossentropy)
-----------------------추가 옵션(안해도 됨) 콜백 (훈련과정 중간에 어떤 작업을 수행하게 해주는 객체)----------------------------
checkpoint_cb=keras.callbacks.ModelCheckpoint('best-model.h5', save_best_only=True)  #모델 훈련 후에 제일좋은 모델 best-model.h5로 저장
early_stopping_cb=keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True) #patience=2 검증점수 향상되지 않아도 2번 기다려줌
-------------------------------------------------------------------------------------------------------------------
#설정해주고 fit할때 callbacks에 넣기
history=model.fit(x_train,y_train,epochs=20,verbose=0 or 1,validation_data=(x_val,y_val),  #0하면 훈련과정 안보임
                  callbacks=[checkpoint_cb,early_stopping_cb])   # .fit에 callbacks옵션에 만들어 둔 객체 넣어줌
# 1.그래프 그리고 최적의 epochs값 결정 or 2.early stopping시 best model불러와 사용 
plt.plot(history.history['loss'])    vs    plt.plot(history.history['val_loss'])  # 각 횟수별 손실
plt.plot(history.history['accuracy'])  vs   plt.plot(history.history['val_accuracy'])
early_stopping_cb.stopped_epoch #으로 어디서 멈췄는지 확인가능 
#베스트모델 불러와 평가 or 콜백 안했으면 그래프 통해 정한 epoch로 학습하고 그 model로 평가 
bestmodel=keras.models.load_model('best-model.h5')
bestmodel.evaluate(x_val,y_val)
bestmodel.predict(x_test[0:5]) -> np.argmax(prediction, axis=1), y_test[0:5]로 비교
#모델 저장
1. model.save_weights('이름.weights.h5')  #모델의 가중치만 저장/ 구조 없음
2. model.save('이름.h5')  #모델의 구조+ 가중치+옵티마이저+손실함수+metircs저장 
# 저장한 모델 불러오기
1. model구조생성-> model.load_weights('이름.weights.h5') -> fit없이 바로 predict가능
2. model=keras.models.load_model('이름.h5') -> 바로 predict, evaluate



1-2회귀 버전
data=pd.read_csv('../machine_learning/perch_3v.csv') #변수 3개
-> poly+minmaxscale화(relu나 sigmoid가 0~1값을 가지므로) / y도 스케일링 필요
model=keras.Sequential()
model.add(keras.layers.Dense(100,activation='relu',input_shape=(10,))) #poly시 변수10개됨, relu
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(10,activation='relu')) #relu
model.add(keras.layers.Dense(1,activation='linear')) # 출력층은 linear 
model.compile(optimizer='adam',loss='mse',metrics=['mae'])
데이터 예측시 그 예측값을 mm.inverse_transform함
model.evaluate에서 mae는 값이 작을수록 좋음

1-3 분류 다른 예시 
fish=pd.read_csv('../machine_learning/fish.csv')
x: minmaxscaling, y: labelencoding
샘플수가 적을경우 dropout해제, early stopping해제, 층여러개, epoch많이 



2. CNN(합성곱층) 
입력->패딩->필터의 커널이 입력에 가해짐->특성맵생성->풀링으로 축소->1차원화->밀집층 
샘플하나당 입력차원배열이 n차원 배열이면 필터도 n차원 배열, 필터하나 찍을때마다 스칼라값 하나 
입력층크기의 (a,b) = 특성맵 크기의 (a,b,n)  
필터의 개수 n = 뉴런의 개수 n = 특성맵 크기( , , n)  
stride=(a,b)에서 a,b가 많이 차이나는 경우는 이미지의 가로세로 크기가 많이 다를때
이미지에서 depth=시간, height=세로픽셀, width=가로필셀, channels=RGB or Black/White

합성곱층과 예시
--conv1d: 텍스트, 주식 데이터등의 용도
    데이터 전처리크기=(batch_size,timesteps,channels)
    input_shape -> model.add(Conv1D(filters=32, kernel_size=3, input_shape=(timesteps,channels))
        !!! 결국 100행(샘플수) 5열(특성수) 짜리의 (100,5) 데이터는 (100,5,1)로 reshape해서 넣음 !!!
                             
--conv2d: 흑백이미지 or 컬러이미지 
    데이터 전처리크기=(batch_size, height, width, channels) #흑백: channels=1, 컬러:channels=3(rgb)
    input_shape -> model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(행,열,1 or 3))
        !!! 결국 100개샘플의 세로28, 가로28짜리의 (100,28,28)은 (100,28,28,1)로 변환해 넣음 !!!
        !!! 원래 컬러이미지는 (100,28,28,3)의 형태라 변환필요X  커널사이즈도 (3,3)그대로 쓰면 알아서 (3,3,3)됨->가중치는 3*3*3개!!!

--conv3d: 영상데이터, 3D 의료영상
    데이터 전처리크기=(batch_size, depth, height, width, channels)
    input_shape -> model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), input_shape=(깊이(시간), 행, 열,3))) 
        !!! 흑백영상 하나받은면 주로 (3600,28,28,1) 이면 영상 하나이므로 (1,3600,28,28,1) 로 변환!!!
        !!! 컬러영상 100개 받으면 (100,3600,28,28,3)이어서 변환필요 X  또한 커널사이즈도 (3,3,3)하면 자동으로 (3,3,3,3)-> 가중치 3*3*3*3개 !!!

--층 여러개 넣을때 1층이 필터 10개-> 2층이 필터 100개면: 2층 지나면 특성맵 100개(28,28,100) !!!!!!!!!!!
  -이는 2층에서는 1층으로 부터 나온 10개의 특성맵 전체를 하나의 입력으로 받음
  : 따라서 2층에 kernel_size=(3,3)이라 적지만 실제로는 (3,3,10(1층 필터개수))으로 채널 자동으로 채워 변환하는 구조 


@@@코드  
데이터: (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
-> x_train.reshape(-1, 28, 28, 1)하고 255로 나누기하고 validation 세트 20%로 나누기 
# 컬러이미지는 이미 (-1,28,28,3)의 형태
model = keras.Sequential()
model.add(keras.layers.Conv2D(32(filter개수), kernel_size=3(커널개수->자동으로 3X3),activation='relu',padding='same',input_shape=(28,28,1)),strides=1 or 2) # 3=(3,3)
model.add(keras.layers.MaxPooling2D(2((n,n)의 풀링필터)) or AveragePooling2D(2) #stride와 padding옵션은 2설정 순간 자동으로 설정돼서 안써도 됨
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')) #두번쨰 CNN층
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())  #일렬로
model.add(keras.layers.Dense(100, activation='relu'))   
model.add(keras.layers.Dropout(0.4))    
model.add(keras.layers.Dense(10, activation='softmax'))          

model.summary() 로 특성맵 크기와 가중치 개수 계산해보기
keras.utils.plot_model(model, show_shapes=True) #이미지로

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.keras',save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)

history = model.fit(x_train,y_train, epochs=20, validation_data=(x_val, y_val),callbacks=[checkpoint_cb, early_stopping_cb])     
plt.plot(history.history['loss']) -> plt.plot(history.history['val_loss']) 로 손실그래프 

model.evaluate(x_val, y_val) # restore_best_weights=True이므로 최적의 파라미터가 자동으로 모델에 설정됨->바로 evaluate, predict하면 됨
preds=model.predict(x_val[0:1]) # [0]이면 (28,28,1) vs [0:1]이면 (1,28,28,1)
classes[np.argmax(preds)]로 클래스 정보 
model.evaluate(x_test,y_test) 마지막으로 test세트 검증




--가중치 시각화 -----------
weights= model.layers[k].weights[0].numpy() #k번째 층의 절편제외한 가중치만을 넘파이배열로 (3,3,1,32): 커널사이즈 (3,3),1,필터개수(32)
  #-histogram: plt.hist(weights.reshape(-1, 1)) 
  #-시각화:
fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):    # 16개 이미지 2줄로
        axs[i, j].imshow(weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)   #(3,3,1,1)에서 (3,3,1,32)까지
        axs[i, j].axis('off')                              #vmin과 vmax로 컬러맵으로 표현할 최소최대값 지정

plt.show() 

--각 필터별 특성맵 시각화: 필터32개면 특성맵 32개------------
#API기능 
model.input으로 입력값 가져올수있음
model.layers[n].output으로 n번째 층의 output결과(z값 배열이나 특성맵) 가져올 수 있음
  -> keras.Model(model.inputs,  model.layers[n].output) # 입력~n번째 층까지만 가지는 모델 생성 
# 하나의 샘플에 대한 n번째 층의 필터 특성맵 시각화하기
fm_nth=keras.Model(model.inputs, model.layers[n].output) #이때 0이면 첫번째 합성곱층, 2해야 풀링층 다음의 두번 함성곱층
sample = x_train[k:k+1].reshape(-1, 28, 28, 1)/255.0 #k번째 샢믈 가져오기
fm_n_k=fm_nth.predict(sample) # predict하면 특성맵값이 출력 (1,28,28,32) 하나의 샘플, (28,28)의 특성맵 32개 
fig, axs = plt.subplots(4, 8, figsize=(15,8))   #n번째 층의 필터개수를 subplot의 가로세로개수로 (64면 8*8) 
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(fm_n_k[0,:,:,i*8 + j]) #하나 샘플의 (28,28)을 전부 가져오겠다
        axs[i, j].axis('off')
plt.show()
