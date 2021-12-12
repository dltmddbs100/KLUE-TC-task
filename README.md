[![main](https://user-images.githubusercontent.com/55730591/145703432-3c949e4a-6da1-4d28-9144-ddf3030b1128.png)](https://www.dacon.io/competitions/official/235747/overview/description)

기간 : 2021.06.30 ~ 2021.08.09 

멤버 : 이승윤(kerry)

결과 : 3 / 647 - 최종 3위

 한국의 kaggle이라고 불리는 AI 경진대회인 Dacon에서 주최한 뉴스 토픽 분류 AI 경진대회에서 최종 3위를 기록했습니다. 이 대회는 **KLUE(Korean Language Understanding Evaluation)** 에서 제공하는 8가지 task 중 TC(Topic classification)에 해당하는 ynat data를 사용한 대회로 총 7가지의 target value를 갖고있습니다. 데이터는 Naver 연합뉴스 기사 제목으로 구성되어있으며 전체 데이터는 약 6만건으로 구성되어있는 text data에 해당합니다. 

 평가 방식은 kaggle과 유사합니다. 제공된 train, test set을 사용하여 test set을 예측하여 Accuracy를 평가합니다. 이는 Public score에 해당하며, 대회가 종료된 후 별도의 private test set을 사용해 재평가하는 과정을 거칩니다. 최종 결과는 대회에서 제공되지 않은 이 private test set을 이용해 평가되며 대회 종료 후 최종 score와 순위가 공개됩니다.

 최종적으로 저는 해당 대회에서 3위를 기록하였습니다. 이를 바탕으로 여기서 대회에 참여한 약 한달간의 분석 과정 및 후기를 공유합니다. 
<br/>
<br/>

### **Instruction**

---

  이 파트에서는 모델 선정과정과 분석에 고심한 부분들을 적고자합니다.

 대회 초반에는 우선적으로 base line model을 선정하고자했습니다. 모델선정에 있어서 최우선적으로 가장 좋다고 알려져있는 Pretrained-BERT모델을 시도해보았습니다. KLUE data는 한국어로 구성된 데이터로, 대부분의 pretrained model은 한국어가 아닌 언어로 구성되어있습니다. 따라서 시도할 수 있는 모델은 제한적이었고 제일먼저 huggingface에서 제공하는 mutilingual bert를 시도해보았으나, score가 낮게 나왔습니다. 

 이후 SKT에서 제공하는 kobert를 사용했는데 한 분이 hugginface의 transformer에서 바로 사용할 수 있도록 설계해놓으셔서 편리하게 사용할 수 있었습니다([https://github.com/monologg/KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)). 해당 모델은 mulitingual bert보다 월등히 높은 score를 보여주었고, 해당 모델을 base line으로 설정하고 전처리 및 hyper-parameter tuning을 진행했습니다. 

 먼저, 다양한 hyper-parameter 조합을 시도하는 과정에서 모델을 학습시킬때 과적합의 양상이 적은 epochs만에 쉽게 나타나는 것을 확인했습니다. 이를 규제하기 위해 다양한 시도 끝에 Dropout를 활용하였고 learning rate를 대폭 줄였습니다. Optimzer는 Adamweightdecay를 활용해 weightdecay를 적용했습니다. 이러한 hyper-parameter의 조정만으로도 상당히 상위권 score에 안착할 수 있었습니다.

 다음으로 전처리 부분입니다. 일반적으로 텍스트 데이터의 전처리는 특수문자 제거, 맞춤법 교정, 유의미한 명사 추출 등으로 이루어집니다. 하지만 이 역시 성능을 보장해주지는 않는데, 이 경우가 그에 해당했습니다. 뉴스 제목임에도 불구하고 상당히 많은 특수문자, 띄어쓰기 오류들이 포함되어있었는데 이를 정제하고자 py-handspell library를 사용했습니다. 그럼에도 불구하고 이전보다 오히려 성능은 크게 낮아지는 모습을 보였습니다. 특히 특수문자를 제거했을때 성능하락의 폭은 더 컸는데, 이는 아무래도 모델이 학습하는 과정에서 특수문자와 target value간의 연관성을 찾아낸 것이 아닌가 싶어 이후에는 특수문자를 완전히 제거하지는 않았습니다.

 해당 방법들로 어느정도 만족할만한 score가 나왔지만 추가적인 향상을 만드는 것은 매우 어려웠습니다. 이러한 상황에서 저는 두가지에 주목했습니다. **바로 large model의 탐색과 data augmentation입니다.** 
<br/>
<br/>


### **Main technique**

---

**< Model >**

 기존의 kobert를 사용한 base model에서 벗어나 다양한 모델들을 시도해보고자 했습니다. 사실 base line 모델 산정 부분 부터 신경쓰이던 부분이 있었는데, 바로 pretrained model의 vocab size였습니다. SKT-KoBERT는 korean-wiki를 기반으로 학습한 모델로,  8,002개의 vocab size를 갖고있습니다. Pretrained Tokenizer로 tokenizing을 시행해본 결과 문제점이 있었는데, **vocab size가 작다보니 대다수의 유의한 단어들을 catch하지 못한다는 점과 wiki만을 기반으로 학습했다보니 news text와는 그 내용이 많이 다르다는 점**입니다. 따라서 폭 넓은 vocab를 갖고있는 한국어 모델들을 우선적으로 찾아보았습니다. 

 그 과정에서 KoElectra, KcBERT 등을 시도해보았으나, 개선되지 않았습니다. XLNET, T5Encoder model, xlm-roberta 등의 모델들을 사용해본적은 없었으나 huggingface 공식문서를 뒤져가며 다양하게 학습을 시도했으나 오히려 KoBERT보다 좋지 못했습니다. 

 이러한 과정을 반복하던 중 최종적으로 찾은 모델은 klue-roberta-model입니다. 언급했듯이 KLUE는 최초의 Korean Bench Mark Dataset으로 공식 github와 그에 대한 논문이 존재합니다. 확인해본 결과, klue-roberta-model은 더 많은 데이터를 기반으로 학습하였으며 vocab size도 32,000여개로 충분했습니다. 대회가 몇일 남지 않은 시점에서, 이를 통해 급격한 성능의 향상을 가져올 수 있었습니다.

<br/>

**< Data Augmentation >**

 대회를 진행함에 있어서 가장 고민을 많이하고 심혈을 기울였던 부분입니다. 가장 핵심적인 부분이며 최종 score에서 큰 효과를 얻었습니다.

 분석에 있어서 데이터의 중요성은 매우 높습니다. **좋은 모델을 만드는 것보다 더 잘 가공된 데이터를 사용하는 것이 성능에 더 유의미한 기여**를 한다는 것은 널리 알려진 사실이며 여러 논문에서도 이를 보여주고 있습니다. 이를 '**Data Centric ML**' 이라고 부릅니다. 따라서 **좋은 모델의 선정과 함께 효과적인 DA technique을 적용하고자 했습니다.**

 DA를 위해 여러가지 논문 및 코드들을 서칭하였고 특히 text classification task에 적용할 수 있는 것들은 많지 않았습니다(주로 Machine Translation에서 사용되는 경우가 많습니다). 이 과정에서 제가 찾은 여러가지 DA 기법들입니다.

> EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks
> [https://arxiv.org/abs/1901.11196](https://arxiv.org/abs/1901.11196)

> Data augmentation Toolkit for Korean text  
> [https://github.com/jucho2725/ktextaug](https://github.com/jucho2725/ktextaug)

> Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations  
> [https://aclanthology.org/N18-2072.pdf](https://aclanthology.org/N18-2072.pdf)

> Pre-trained Data Augmentation for Text Classification  
> [https://github.com/hugoabonizio/predator](https://github.com/hugoabonizio/predator)

 이 중 대표적으로 널리 알려진 EDA는 쉽고 편리한 장점이 있습니다. 하지만 이 역시 대상 언어의 wordnet을 필요로합니다. 한국어로 구축된 wordnet을 활용해 EDA를 적용해 oversampling했을때, 오히려 추가된 noise들과 부정확한 wordnet으로 인한 동의어의 대체가 이루어지다보니 성능이 저하되는 현상이 나타났습니다. 다음으로 적용한 ktextaug는 왜인지는 알 수 없으나 일정 용량이상의 경우부터 실행되지않는 모습을 보여 약 4만여개에 달하는 학습데이터에 적용할수가 없었습니다. 그 외에도 다양한 기법들이 존재했으나, 한국어에 맞는 사전을 직접구축해야하는 등 시간과 비용의 제약이 많아 현실적으로 사용은 불가능했습니다.

 기억을 되짚어보면 DA를 찾아보고 적용하는데 정말 많은 시간을 들였던 것 같습니다. 아무리 다른 방법을 생각해보아도 DA에 답이있다라는 강한 확신이있었기 때문에 계속해서 고심했습니다. 그러던 중 Machine Translation분야에 관해 읽었던 논문 중 하나가 떠올랐는데, **바로 Back Translation입니다.**

> **Understanding Back-Translation at Scale**  
> [https://arxiv.org/abs/1808.09381](https://arxiv.org/abs/1808.09381)

#####※ 위 논문은 BT를 소개한 논문이 아닌 효과적인 BT의 적용을 위한 조건들을 실험한 결과를 보여주는 논문입니다.

 간단히 Back Translation을 소개하자면, **BT는 단일 corpus로부터 번역을 통해 데이터를 증강하는 일종의 DA technique**입니다. 사실 이 방법은 기계번역에서 고안되어 번역 성능을 높인 동시에, bilingual corpus가 아닌 monolingual corpus로부터 데이터를 증강하는 기법으로 소개되었습니다. 

 예를들어, 한-영 번역기를 만들고자할때 우리는 그에 필요한 한국어-영어 pair data가 필요합니다. 이렇게 완벽하게 매칭되어있는 데이터는 수집이 어렵지만 단일 언어로 작성된 monolingual corpus는 그에 비해 수집이 용이합니다. BT는 이부분에 주목합니다. 한국어-영어 pair data가 있다면 한-영 번역기가 아닌 반대로 영-한 번역기를 만들 수 있습니다. 따라서 영한 번역기를 먼저 구축해 monolingual corpus에 해당하는 영어를 영한 번역기에 투입하면 그에대한 output으로 한국어 corpus를 얻을 수 있습니다. 이렇게 얻은 새로운 pair를 이용해, 원래목적인 한-영 번역기의 학습에 사용하는 기법입니다.

 해당 방법을 사용하려면 번역기의 구축이 필요합니다. 하지만 분류문제를 위해 새로운 번역 모델을 만드는 것은 매우 비효율적인 일입니다. 따라서 기존에 구축되어 있는 번역기를 활용했습니다. 방법은 다음과 같습니다.

![bt](https://user-images.githubusercontent.com/55730591/145703488-67abf180-9b26-4459-ac4e-75038097b975.png)

####1.  **구축된 번역기를 통해 train data의 text를 타 언어로 번역합니다.**
####2.  **타 언어로 번역된 text를 다시 한국어로 번역합니다.**
####3.  **앞선 두번의 번역을 거쳐 생성된 새로운 문장을 train data에 추가하며, 평가를 위한 validation set은 변형된 문장을 포함하지 않습니다.**

 위 방법을 이용하여 기존 text 만큼의 문장들을 더 얻을 수 있었습니다. 두 번의 번역을 거치면서 원본 text는 조금씩 변형됩니다. 유사한 의미를 가진 동의어로 단어가 대체되기도하고, 순서가 바뀌기도하며 오히려 조금은 부자연스러운 문장들이 형성됩니다. 이렇게 **생성된 어색한 문장은 학습시 강한 train signal을 발생시키기 때문에 과적합을 줄이거나 예측력을 높이는 것에는 도움이 됩니다.** 다만 주의해야할 점은, 검증을 위한 validation set에는 변형된 문장을 포함시키면 안된다는 점입니다.

 대회가 종료된 후, Private 최종 score가 공개되었습니다. 제출한 두 가지 파일은 다른 모든 조건이 동일할때, BT를 적용하지 않은 경우와 BT를 활용한 경우의 submission입니다. 결론적으로 BT를 적용하지 않은 경우는 약 20위권에 해당하는 score로 공개된 test set에 대한 과적합이 더 나타났으며 BT를 적용한 score를 통해 3등에 위치할 수 있었습니다.

 여기까지는 BT에 대한 전반적인 활용 방안을 말씀드렸다면, 부가적으로 실제 데이터에 있어서는 어떤방식으로 적용했는지와 실제로 적용하면서 생긴 문제점들에 대해 고심했던 점들을 구체적으로 설명해드리고자합니다.

<br/>

**< More details >**

  BT를 활용하기 위해서는 구축되어있는 번역기가 필요합니다. 대표적으로 Papago, google, kakao i 등의 범용적으로 사용되는 번역기가 존재합니다. 결론부터 말씀드리자면 저는 두 가지 번역기를 사용했습니다. Papago와 kakao brain에서 제공하는 Pororo 입니다. 간략하게 선택한 이유를 말씀드리자면, 여러가지 문장들을 번역기에 넣고 비교해본 결과 Papago의 번역성능이 더 높다고 판단했으며, 원문의 특수문자의 위치나 문장안에있는 모종의 규칙들을 가장 잘 보존하는 것으로 보였기 때문입니다.

 하지만 처음부터 문제점이 존재했는데, Papago는 네이버에서 제공하는 API를 활용했을때 하루에 10,000단어씩만 번역을 무료로 사용할 수 있다는 점입니다. 대회에서 사용하는 train data는 약 45,000 문장으로 실질적으로 API를 통한 번역은 불가능했습니다. 그래서 결국 Crawling을 활용하기로했습니다. Crawling을 활용하면 번역횟수에 제한은 없으나, 체감상 속도가 느리고 오번역되는 경우가 많았습니다. 사실 만단위의 문장을 번역한다는 것이 절대적으로 시간이 오래걸리는 일인것은 확실합니다. 

 언급한 오번역되는 경우는 여러가지 케이스가 존재했습니다. 지나치게 오번역되는 문장들이 학습시 포함된다면 오히려 학습을 크게 방해하는 요소로 작용할 것이 분명하기때문에 **번역된 문장의 질은 정말 중요했습니다**. 이에 따라 재번역대상의 기준을 정립했습니다.

> 1\. 원본 문장에 한자가 포함되어있는 경우  
> 2\. 번역된 문장의 한글이 차지하는 비중이 0.6이상인 경우  
> 3\. 번역된 문장이 기존 문장의 길이에 대한 비율이 0.5이하인 경우  
> 4\. 번역이 이뤄지지 않아 null값을 갖는 경우

 위와 같은 기준으로 재번역대상을 정의하고 Papago로 재번역한뒤, 그럼에도 불구하고 제대로 번역되지 않는 문장들은 kakao brain에서 제공하는 Pororo를 이용해 번역하여 모든 train set에 대한 번역을 완료했습니다.

![case](https://user-images.githubusercontent.com/55730591/145703501-560ac6a6-b84d-4bc8-a05b-b3f577363e37.png)

  결과적으로 의미는 유사하지만 사용되는 단어 종류나 위치가 다른 새로운 문장들이 생성되었습니다. 그러나 한 가지 치명적인 문제에 직면했는데, 이는 '문장의 결' 즉 뉘앙스가 다르다는 점입니다. 생성된 문장들은 대체적으로 원본 문장에비해 길이가 아주 긴 특징을 가졌습니다. 원본 뉴스 문장은 주로 명사로 끝나거나 '..임' , '..했음', '..로 보여' 등 문장이 완성되어있지 않은 경우가 많은 반면, 번역된 문장은 대부분 '..했다', '..합니다' 등으로 완성형 문장으로 끝나기 때문에 길이가 길어진 것입니다. 또한 원문에서는 사용되지 않은 특수문자들도 포함되었고 특수문자가 사용되는 규칙도 달랐습니다.

 예를들면 원본 뉴스 문장은 '…'이 포함된다면 대부분 문장 중간에 사용되며 후미에는 '...'를 사용합니다. 단순히 특수문자를 제거할 수 도 있지만, 앞선내용에서 특수문자를 제거했을때 우리가 알지못하는 일정한 규칙의 존재로 인해 성능이 떨어지는 것을 보았기 때문에 이러한 특수문자들의 사용 규칙과 문장의 결을 일치시켜주는 것은 중요한 일이라고 생각했습니다. 언뜻보기에도 특수문자들이 사용되는 규칙은 상당히 많아보였고 끝나는 어미의 차이가 매우 컸기때문에 이러한 '문장의 결'을 맞춰주는 작업을 정규표현식을 이용해 수동으로 제거하는 과정을 거쳤습니다.

비록 완벽하게 원본 문장이 가지는 규칙이나 뉘앙스와 일치시키는 것은 불가능했지만, 유의할 것으로 판단되는 대부분의 경우들을 고려해 정제를 마쳤습니다.
<br/>
<br/>


### **What else?**

---

 제가 대회에서 사용한 주요 테크닉은 위에서 제시한 것들이지만, 그 외에도 부가적으로 고려한 부분도 간략히 정리해보고자합니다. 

**< Text Similarity >** 

 유사도가 높은 문장들을 추출하는 과정에서 발견한 것으로, 라벨링 과정에서 생긴 오류로 보여집니다.

![case2](https://user-images.githubusercontent.com/55730591/145703647-dc5ff010-4df5-414e-95f5-dd232d9b73cf.png)

 분명 의미가 완전히 동일한 문장이지만 서로 가지는 target value가 다릅니다. 이러한 문장 pair들이 약 300여건이 넘게 있는 것을 확인할 수 있었습니다. 실제상황이라면 이러한 라벨들은 일정한 기준으로 교정하는 것이 옳습니다. 하지만 대회상황이라면, 평가받아야하는 미지의 test set에 이러한 이상치가 포함되지 않는다는 보장이 없습니다. 따라서 선택을 해야했는데, 저는 이 문장들을 반드시 학습시에 포함시켰습니다. 어쩌면 학습에 방해가될 수도 있지만, 모델이 스스로 이들간의 미세한 차이를 찾을 것이라고 생각했기 때문입니다. 또한 실험적으로도 이들을 고려하지 않는 것보다 학습시에 포함하는 것이 더 좋은 결과를 가져오는 것을 확인했습니다.
<br/>

**< Stratified K-fold Average >**

 교차검증을 의미하는, 흔히 CV라고 부르는 검증전략의 선택은 매우 중요합니다. 여기서는 target value가 총 7개로 확인해본결과 클래스가 균등하지 않은, Class imbalanced problem에 해당합니다. 따라서 검증 전략에 있어서 Stratified K-fold가 적합하다고 판단했으며 모델 학습시 5-fold를 사용해 각 fold에 대해 학습하고, test set에 대해 예측한 probability들을 average하는 일종의 단일모델 stacking으로써 활용하였습니다. 특출난 방법은 아니지만, 해당 전략을 통해 유의미한 성능의 개선이 이루어지는 것을 보고 이후 모든 모델에 해당 전략을 사용했습니다.
<br/>
<br/>


### **End**

---

 대회를 진행하며 여러가지 우여곡절을 겪었습니다. 현역 군인신분이다보니 Google Colab을 사용할 수밖에없었는데 Runtime Error로 인해 실행했던 작업이 끊겨서 다시해야하는 상황도 직면하고 Crawling을 하다가 알 수 없는 오류로 인해 다시해야하는 경우도 생기고 OOM(Out of memory) error도 자주 겪었습니다. 특히 이런 환경적인 제약때문에 GPU에 대한 갈망이 매우 커진대회였습니다. 약 25일정도동안 집중해서 진행한 것 같은데 다행히 좋은 결과가 있어서 기쁩니다. 이번 NLP 대회를 통해 BERT를 포함한 수많은 모델들과 여러가지 DA기법들을 탐색하면서 많은 시간을 보내다보니 전부 제게 큰 자산이 된 것 같습니다. 사용한 코드와 관련 출처들을 밝히며 글을 마치겠습니다.
<br/>
<br/>


#### **Reference**

-   DACON Code - [https://dacon.io/competitions/official/235747/codeshare/3054?page=1&dtype=recent](https://dacon.io/competitions/official/235747/codeshare/3054?page=1&dtype=recent)
-   KLUE github - [https://github.com/KLUE-benchmark/KLUE](https://github.com/KLUE-benchmark/KLUE)
-   Back Translation - [https://deepkerry.tistory.com/17](https://deepkerry.tistory.com/17)
