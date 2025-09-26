#제일 기본 모델 TFGPT2Model(TFbertModel과 유사)
model=TFGPT2Model.from_pretrained('')
output=model(x)

나오는 값은 모든 입력 토큰에 대한 출력값(output.last_hidden_state)
->이때 각 토큰은 768(1028등도 가능)차원의 값을 가짐

#분류할시
output.last_hidden_state에서 
  1. mean이던 max로든 pooling함 (tf.reduce_mean(outputs.last_hidden_state, axis=1))
  2. 마지막 토큰만 사용(마지막 토큰에서 그전까지 생성된 모든 토큰에 대한 요약본 담고있음: outputs.last_hidden_state[:, -1, :])-> bert의 cls처럼 사용
이후 dense층으로 들어가 분류개수만큼 뉴런설정

<예시>
문장 ("I love pizza")
   ↓ tokenizer
["I", "love", "pizza"] → 토큰 3개
   ↓ GPT/BERT
last_hidden_state.shape = (batch=1, seq_length=3, hidden_size=768)
   ↓ pooling (예: 평균) or 마지막 토큰
문장 벡터 = (batch=1, hidden_size=768)
   ↓ Dense(num_labels=2)
출력 = [0.1, 0.9]  (부정=0.1, 긍정=0.9)


#생성형의 경우
outputs.last_hidden_state[:, -1, :]로 마지막 토큰 출력값 가져옴(전의 모든 맥락정보를 가지고 있음)
이 값을 전체 vocab_size수 만큼의 뉴런을 가진 dense층에 넣어 소프트맥스로 이어질 단어 중 제일 확률높은 단어 가져옴 by argmax
-> 이 과정을 다시 반복해 뒤에 이어질 단어 또 가져옴  

<예시>
문장 "I love"
 → 토큰 ["I", "love"]
 → last_hidden_state: (batch=1, seq_length=2, hidden_size=768)
 → 마지막 벡터 → Dense(vocab_size=50k)  # 모든 단어사전
 → 다음 단어 분포 ("you":42%, "pizza":25%, ...) ->'you'선택
->다시 'i love you'문장으로 입력하여 그 다음 단어 예측
 → 반복하면 문장이 생성됨 
