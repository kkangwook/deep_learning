import tensorflow as tf

# np.array와 유사한 기능
tf.constant([1,2,3,4,5],dtype='float') #1차원 벡터

# 연산 가능
x = tf.constant([1,2,3,4,5], dtype="float") # Tensor(1차원) 
y = tf.constant([2,4,6,8,10], dtype="float") # Tensor(1차원) 
z = x + y # 사칙연산 : 텐서 연산  
print('z =', z) # z = tf.Tensor([ 2.  4.  6.  8. 10.], shape=(5,), dtype=float32) : 텐서 정보
