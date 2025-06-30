import tensorflow as tf

#########################
텐서에 일반 연산자 사용 -> 결과:텐서
넘파이에 텐서함수 사용 -> 결과: 텐서
#########################


# np.array와 유사한 기능
tf.constant([1,2,3,4,5],dtype='float') #1차원 벡터

# 연산 가능
x = tf.constant([1,2,3,4,5], dtype="float") # Tensor(1차원) 
y = tf.constant([2,4,6,8,10], dtype="float") # Tensor(1차원) 
z = x + y # 사칙연산 : 텐서 연산  
print('z =', z) # z = tf.Tensor([ 2.  4.  6.  8. 10.], shape=(5,), dtype=float32) : 텐서 정보


### TensorFlow는 NumPy 코드처럼도 쓸 수 있지만, 실제 성능을 끌어내려면 @tf.function으로 그래프화해서 쓰는게 좋음
# 1. 즉시실행 모드(기본) : 인터프리터 방식 -> 데이터 소량일 경우
def add_eager_mode(x, y): # Python 함수
    #print(tf.executing_eagerly()) # True : 즉시실행 모드 on
    return tf.add(x, y) # Python자료 즉시 실행 & 그래프(x)  

# 2. 데코레이터로 그래프 모드: 대용량일떄
@tf.function # 함수 장식 : 그래프(Graph) 모드 지원 -> 더 좋은 성능 
def add_graph_mode(x, y): # Python 함수
    # print(tf.executing_eagerly()) # False면 그래프 모드다
    return tf.add(x, y) 

print(add_graph_mode(x, y)) #tf.Tensor([ 2.  4.  6.  8. 10.], shape=(5,), dtype=float32)


# tf결과를 numpy형태로
result_tensor.numpy() # 결과에 .numpy() 

# numpy를 tf.add와 같은 함수에 넣어도 됨

# 그래프 구조
노드(점): 값, 연산(+,-), 함수(relu)등 가능
엣지(선): 텐서데이터 흐름


# 그래프 시각화하기
dir = "C:/ITWILL/logs/calc/" + ***디렉토리_이름 정해주기***
writer = tf.summary.create_file_writer(dir) #파일 저장 객체와 함수
tf.summary.trace_on(graph=True, profiler=False)  
with writer.as_default(): # 블록 안에서 실행되는 연산 그래프를 로그파일에 기록 
    # Tensor 생성(데이터)  
    x = tf.constant(1.0, dtype=tf.float32) # 텐서 상수  
    y = tf.constant(2.0, dtype=tf.float32) # 텐서 상수   
    # 함수 호출 & Graph 로그파일  
    result = ***함수***(x, y) # 여기에 함수    
    tf.summary.trace_export(name="ArithmeticGraph", step=0) 
print(result) #하면 값나옴
# anaconda prompt에서 한줄씩 입력해 tensorboard 오픈
conda activate tensorflow
tensorboard --logdir=C:\ITWILL\logs\calc\***아까 정해준 이름***
-> 뜬 사이트 ctrl+클릭으로 사이트 오픈 -> 시각화된 정보 볼수있음(노도와 엣지로 이루어진)
    -> input, parameter, pred, loss, optimizer등 존재



#### Tensor : 상수 또는 다차원 배열
scala = tf.constant(1234) # 0차원(상수)  
vector = tf.constant([1,2,3,4,5]) # 1차원 
matrix = tf.constant([ [1,2,3], [4,5,6] ]) # 2차원
cube = tf.constant([[ [1,2,3], [4,5,6], [7,8,9] ]]) # 3차원 or 3-tensor
n차원은 rank=n, shape=[d0,d1,d2,...,dn-1], N-tensor라 부름

# tensor속성
tf.rank(x) # 차원수: 위의 값 각각 0,1,2,3
tf.shape(x) # 모양: (), (5,), (2,3), (1,3,3)
tf.size(x) # 원소개수: 1,5,6,9

x.dtype
x.get_shape() or x.shape
x.ndim
x.numpy()


# 수학관련 함수
tf.math.add() # 덧셈 함수
tf.math.subtract() # 뺄셈 함수
tf.math.multiply() # 곱셈 함수
tf.math.divide() # 나눗셈(몫 계산) 함수
tf.math.mod() # 나눗셈(나머지 계산) 함수
tf.math.abs() # 절댓값 변경 함수
tf.math.square() # 제곱 계산 함수
tf.math.sqrt() # 제곱근 계산 함수
tf.math.round() # 반올림 함수
tf.math.pow() # 거듭제곱 함수
tf.math.exp() # 지수 함수
tf.math.log() # 로그 함수

# numpy 유사 함수
tf.reduce_sum(data,axis=1]) # np.sum
x.get_shape() # x.shape
tf.matmul(x, y) # @ 행렬곱
x[0,0], x[:,0], x[0,:] # 인덱싱 가능

#모양변경하기
tf.reshape(data, (2, 3)) # np.reshape
tf.transpose(x) # (2,3)->(3,2)
tf.squeeze(x) # 차원size 1인경우 제거 ex) (1,2,1,3)->(2,3) 
tf.expand_dims(x, axis=n) # 한 차원 늘려줌


### 텐서플로우 데이터 생성하기
tf.constant([1, 2, 3], dtype=tf.int32) #np.array와 유사-> 값 못바꿈
tf.Variable([1.0, 2.0, 3.0], dtype=tf.float32) # tf.constant와 유사하지만 나중에 값 바뀔수있음
    .assign([4,5,6])으로 variable은 변경가능 
tf.zeros([2, 3])  # 2x3 행렬(0으로 이루어진)
tf.ones([2, 3]) # 2x3 행렬(1로 이루어진)
tf.fill([2, 3], 7) # 7로 이루어진 2X3행렬
tf.range(start, limit=None, delta=1) # 파이썬의 range/ delta는 증가값
tf.linspace(start, stop, num) # num은 사이에 있을 개수 

#랜덤 텐서 생성 ( shape=[x,y]와 같은 형태)
tf.random.normal(shape, mean=0.0, stddev=1.0) #정규분포
tf.random.uniform(shape, minval=0, maxval=None) # 균등분포
tf.random.set_seed(seed) # 시드 설정


## 활성화 함수
#시그모이드
1 / (1 + tf.math.exp(-x))

# 소프트맥스
ex = tf.math.exp(x - x.max())
y = ex / sum(ex)
