doobnet

DOOBNET (Deep Object Occusion Boundary Detection from an Image)

Pascal instance occlusion dataset (PIOD)

In computer vision, the study of occlusion reasoning has been largely confined to the context of stereo, motion and other multi-view imaging problems

The problem of object occlusion boundary detection relies on having precise object boundary

leaving a large room for improvement 개선의 여지가 많다.

f-score 에 대한 설명

https://bcho.tistory.com/tag/F1%20score

https://www.quora.com/What-is-an-intuitive-explanation-of-F-score

segmentation : FCN [19], SegNet [20], U-Net [21]

dilated convolution

깃헙 소스 : https://github.com/GuoxiaWang/DOOBNet

다음 논문은 Doc: Deep occlusion estimation from a single image
를 보고 싶은데 다운받을수가 없네...

통상의 이미지는 1퍼센트의 바운더리 픽셀을 가짐

Class-balanced Cross Entropy
Focal Loss

논문의 중간 부분은 네트웍에 대한 설명(layer 설명)
doobnet은 인코더 디코더 형태로 구성디어있고 
인코더 모듈로 res50을 사용함
네트웍은 그림을 참고하는게 편할거 같다.
네트웍에 대한 설명 이후 는 성능에 대한 리포트...

성능표 볼줄 아시는분??? AP?? ODS?? OIS?? 뭐지?

To address the problem, we propose a discriminating loss function motivated by FL, called the Attention Loss (AL), focusing the attention on class-agnostic object boundaries. Note that the true positive and true negative examples belong to well-classified examples after the loss are balanced by alpha weight and FL can continue to partially solve the class-imbalance problem. However, the number of false negative and false positive examples is small and that their loss are still overwhelmed during training. Meanwhile, training is insufficient and inefficient, leading to degeneration of the model. The attention loss function explicitly up-weights the loss contribution of false negative and false positive examples so that it is more discriminating.
이 문제를 해결하기 위해, 우리는 FL에 의해 동기 부여된 차별화된 손실 기능을 제안하는데, 이것은 등급에 구애받지 않는 물체 경계에 주의를 집중시키는 것이다. 참된 양의 예와 참된 부정적인 예들은 손실이 알파 무게로 균형을 이룬 후에 잘 분류된 예에 속하며 FL은 계속해서 클래스-임밸런스 문제를 부분적으로 해결할 수 있다. 그러나 거짓 음성 및 거짓 양성 예시의 수는 적고 훈련 중 그들의 손실이 여전히 압도되고 있다. 한편, 훈련은 불충분하고 비효율적이어서 모델이 퇴화한다. 주의력 상실 기능은 명백한 상향-허위 음성 및 거짓 양성 예제의 손실 기여를 평가하여 더 차별화한다.



In this paper, we propose the Attention Loss to address the extreme positive/negative class imbalance, which we have suggested it as the primary obstacle in object occlusion boundary detection. We also design a unified end-to-end encoder-decoder structure multi-task object occlusion boundary detection network that simultaneously predicts object boundaries and estimates occlusion orientations. Our approach is simple and highly effective, surpassing the state-of-the-art methods with significant margins. In practice, Attention Loss is not specific to occlusion object boundary detection, and we plan to apply it to other tasks such as semantic edge detection and semantic segmentation in the future. Source code will be released.
본 논문에서는 객체 폐색경계검출의 1차적 장애물로 제시해 온 극도의 양성/음성의 계급 불균형을 해소하기 위해 주의력 상실을 제안한다. 또한 객체 경계를 동시에 예측하고 폐색 방향을 추정하는 통합 엔드투엔드 인코더-디코더 구조 멀티태스크 객체 폐색 경계 탐지 네트워크를 설계한다. 우리의 접근방식은 간단하고 매우 효과적이어서 상당한 마진을 가진 최첨단 방법을 능가한다. 실제로 주의력 상실은 폐색 물체 경계 검출에만 한정되지 않으며, 향후 의미 에지 검출, 의미 분할 등의 다른 작업에 적용할 계획이다. 소스 코드가 공개될 것이다.
