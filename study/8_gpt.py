########################제일 기본 모델 TFGPT2Model(TFbertModel과 유사)
tokenzier=autotokenizer.from_pretrained('')
x=tokenizer(text)['input_ids'] #input_ids만 들어가면 됨(attention_mask는 선택)!!!!!!!!!!!!!!!!!!!1

model=TFGPT2Model.from_pretrained('')
output=model(x)

나오는 값은 모든 입력 토큰에 대한 출력값(output.last_hidden_state)
->이때 각 토큰은 768(1028등도 가능)차원의 값을 가짐

############################분류할시
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


##############################생성형의 경우
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


-1. 그렇다면 학습시킬 데이터 셋 구조는?
문장: "I love pizza very much"에 대해
입력값=[i,love,pizza,very,much] (->tokenizer로 토큰화하면 [i,love,pizza,very,much,eos]로 하나 더 추가)
label(정답값)=[love,pizza,very,much,eos]  (->tokenizer로 토큰화하면 [love,pizza,very,much,eos,ignore]
=>i를 입력하면 정답을 love라고 학습시키고 그다음은 i love를 주면 그 정답값이 pizza라고 학습시키고 이걸 반복해서 i love pizza very much를 입력시켜서 eos가 나오면 끝나는 구조 

-2. 학습시키는 법
input_ids = [I, love, pizza, very, much]
labels    = [love, pizza, very, much, <EOS>]

outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss  # 모델이 내부적으로 모든 위치에 대해 loss 계산
=> 모델은 자동으로 시퀀스 전체를 한 번에 처리 → 각 위치별 logits 계산 → loss 합산
=> 따라서 for문은 필요X

-3. 학습된 모델로 생성시킬때(for문 필요)

prompt = "I love" #내가 여기서 부터 생성하겠다고 내가 정한기준
generated = tokenizer(prompt, return_tensors="tf")["input_ids"] #

for _ in range(max_len):
    outputs = model(generated)
    next_token_logits = outputs.logits[:, -1, :]
    next_token = tf.argmax(next_token_logits, axis=-1)
    generated = tf.concat([generated, next_token], axis=-1) #예측 된 값을 concat으로 옆으로 붙여서 다시 모델에 집어넣음
    if next_token == tokenizer.eos_token_id:
        break

        
######################### Q/A #####################
이것도 생성형이랑 유사-> 대신 [질문+정답]을 한번에 input으로 입력
-> x=질문, y=정답 대신 x=[질문+정답], y=[문+정답+eos]가 더 정확
-> 또한 “질문:” / “답변:” 토큰넣어서 모델이 질문과 답변의 경계를 인지하도록 함

<예시>
질문: 오늘 날씨 어때?
답변: 맑습니다.

-1. 데이터 셋 구조
"질문: 오늘 날씨 어때?\n답변:맑습니다."     #이런식으로 변환, 질문:, 답변: 값도 같이 넣어줌
토큰화시 x = ['질문', ':', ' 오늘', ' 날씨', ' 어', '##때', '?', '\n', '답변', ':', ' 맑습니다', '.'] 로 변형

x = ['질문', ':', ' 오늘', ' 날씨', ' 어', '##때', '?', '\n', '답변', ':', ' 맑습니다', '.']
y = [':', ' 오늘', ' 날씨', ' 어', '##때', '?', '\n', '답변', ':', ' 맑습니다', '.', <EOS>]
 => 이때 반드시 항상 토크나이즈 후 시퀀스를 기준으로 한 칸 오른쪽 shift + EOS!!!!!!!!!!!!!!!!! 



-2. 학습시킬때는 for문X
model.fit(x, y) # 위의 x,y를 사용



-3. 생성할때는 for문 쓰거나 아니면 for문없이 model.generate()사용

prompt = "질문: 오늘 날씨 어때?\n답변:" # 첫 시작 prompt는 "~답변:" 까지 주면 뒤의 정답을 알아서 생성해줌

# 1) 직접 for문으로 토큰 하나씩 생성
generated = tokenizer(prompt, return_tensors="tf")["input_ids"]
for _ in range(max_len):
    outputs = model(generated)
    next_token_logits = outputs.logits[:, -1, :]
    next_token = tf.argmax(next_token_logits, axis=-1)
    generated = tf.concat([generated, next_token], axis=-1)
    if next_token == tokenizer.eos_token_id:
        break

# 2) HuggingFace generate() 활용 (추천) -> for문 쓸 필요 X
input_ids = tokenizer(prompt, return_tensors="tf").input_ids
generated_ids = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(generated_ids[0])
