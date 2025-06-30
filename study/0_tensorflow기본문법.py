import tensorflow as tf

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

# 그래프 구조
노드(점): 값, 연산(+,-), 함수(relu)등 가능
엣지(선): 텐서데이터 흐름
