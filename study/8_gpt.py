                   **gpt구조**

         입력값=(batch_size,seq_len)
                    ㅣ
      -입력임베딩(토큰임베딩+ 위치임베딩)-
    =>출력값(batch_size, seq_len, hidden_size=768)
                    ㅣ
      -transformer decoder block X N-
             -Masked Multi-Head Self-Attention(삼각형의 casual masking적용된)-
             -Residual Connection + LayerNorm(입력 + attention 출력 → 정규화)-
             -Feed-Forward Network (MLP), gelu활성화 사용-
             -Residual Connection + LayerNorm-
     =>출력값(batch_size, seq_len, hidden_size=768)
                    ㅣ
      -LM Head (Dense layer with vocab_size)
    =>출력값(batch_size, seq_len, vocab_size)    
                    ㅣ
                 -softmax-



########################제일 기본 모델 TFGPT2Model(TFbertModel과 유사)
tokenzier=autotokenizer.from_pretrained('')
x=tokenizer(text)['input_ids'] #input_ids만 들어가면 됨(attention_mask는 선택)!!!!!!!!!!!!!!!!!!!1

model=TFGPT2Model.from_pretrained('')
output=model(x)

나오는 값은 모든 입력 토큰에 대한 출력값(output.last_hidden_state)
->이때 각 토큰은 768(1028등도 가능)차원의 값을 가짐

############################ 분류할시 #########################
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


############################## 생성형의 경우 ############################
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
    next_token_logits = outputs.logits[:, -1, :]    ###여기서 마지막 토큰으로 그 다음 토큰 예측 !!!!!!!!!!!!!
    next_token = tf.argmax(next_token_logits, axis=-1)
    generated = tf.concat([generated, next_token], axis=-1) #예측 된 값을 concat으로 옆으로 붙여서 다시 모델에 집어넣음
    if next_token == tokenizer.eos_token_id:
        break


################################## 입력/출력 길이가 다를수있는 이유 ######################
학습시킬떄는 입력값과 label값의 입력크기는 같아야하지만(입력에서 한칸shift하고 eos더한게 label이여서 결국 둘의 길이 같음)
결국 for문이나 generate을 쓰기때문에 생성할떄는 입력과 출력값의 길이가 다를수있음(eos가 오면 break되므로)
또한 gpt학습시킬때 각 배치(Batch) 안 샘플 길이는 같아야 함(패딩 시켜서 길이 맞춤)
###########################################################################################


################################ 층 구조 ####################################
위의 생성형 포함 이 밑에 있을 q&a나 번역 모델들 전부 
input값을 기본 gptmodel넣음 
-> 나온 last_hidden_state값을 dense(voca_size)에 넣음 (학습시킬때는 전부)
-> 나온값(logits)을 input의 shift+eos한 값을 정답으로 해서 학습시키면 됨   #나오게 되는 값은 (batch_size, seq_len, vocab_size)==각 위치(토큰)에 대해 어휘 전체에 대한 확률분포 예측

그리고 생성할때는 for문+ outputs.logits[:, -1, :]을 사용  or generate하면 자동으로 생성
###############################################################################



################################# 손실계산법 ###################################
출력logits = (batch_size, seq_len, vocab_size)   → 각 위치(토큰)에 대해 어휘 전체에 대한 확률분포 예측값
정답labels=(batch_size, seq_len) → 각 위치에서 정답 토큰의 id

loss = mean(cross_entropy(logits, labels)) -> "예측 분포 vs 정답 one-hot 벡터" 비교
-> logits를 소프트맥스화하지 않고 logits의 예측분포값을 labels의 one-hot벡터와 비교
-> 이때 크기가 다르다고 생각할수 있지만 사실은 labels=(batch_size, seq_len)에서 seq_len은 결국 각 값들을 원핫인코딩해서 (batch_size, seq_len, vocab_size)로 해서 같은크기로 비교

<예시> seq_len2짜리 하나의 샘플에 대해
-출력값-
토큰 위치 1 → [0.1, 0.6, 0.1, 0.1, 0.1]
토큰 위치 2 → [0.7, 0.1, 0.1, 0.05, 0.05]

-labels-
[1, 0]  -> [0, 1, 0, 0, 0]
           [1, 0, 0, 0, 0]

-crossentropy계산-
위치1 손실 = -log(0.6)
위치2 손실 = -log(0.7)
최종 손실 = 평균(-log(0.6), -log(0.7)) 
=> 모든 샘플의 모든 위치의 손실값을 평균내서 최종 스칼라 loss 값 1개로 학습

##
sparse_categorical_crossentropy(from_logits=True)를 쓰면:
logits는 그대로 넣음 (softmax 안 해도 됨, 내부에서 처리)
labels는 index만 있어도 됨 → 내부적으로 one-hot으로 변환 후 cross entropy 계산

###########################################################################


######################### 평가법 ############################
분류와 달리 생성형은 하나의 샘플 내부에서 조차 각 위치별 다음 토큰 예측 학습을 수행하기 땜에 하나의 샘플에서는 o/x가 아닌 내부에서 얼만큼 맞았는지 정확도 값으로 나옴(ex:67%)
-> 그래서 그 샘플이 정확히 맞으면 1 아니면 0 이런식의 정확도가 아닌 
-> 모든 샘플의 (맞춘 토큰 수 / 전체 토큰 수) → 평균  !!!!!!
-> 또한 단순 정확도 뿐만 아니라 "사람이 읽었을 때 얼마나 자연스러운가?"를 보기 위해 BLEU, ROUGE 같은 지표를 씀
##############################################################




        
#########################  Q/A  ###################
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




######################### 번역 #########################
-> 생성형, qa와 유사하다보면 됌
-> 대신 "질문:, 답변:" 말고 "translate English to Korean:"와 "->"를 넣어줌

<예시>
source (영어): "I love pizza."
target (한국어): "나는 피자를 좋아해."

-1. 데이터 셋 구조
->시퀀스연결: "translate English to Korean: I love pizza. → 나는 피자를 좋아해."
(이때 '->'나 '\n'넣어줘도 되고 안넣어줘도 됨  => 짜피 모델이 알아서 경계를 구분함)

x = tokenizer("translate English to Korean: I love pizza. 나는 피자를 좋아해.")  # input_ids
y = x shifted right + EOS



-2. 학습
model.fit(x,y)


-3. 생성(for문 쓰거나 generate)
prompt = "translate English to Korean: I love pizza."
input_ids = tokenizer(prompt, return_tensors="tf").input_ids

generated_ids = model.generate(input_ids, max_length=50)
translated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(translated_text)  # "나는 피자를 좋아해."




