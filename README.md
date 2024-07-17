# BeautyGAN

### 소개

BeautyGAN: 인스턴스 수준의 얼굴 메이크업 전환을 위한 딥 생성적 적대 신경망

공식 웹사이트: [http://liusi-group.com/projects/BeautyGAN](http://liusi-group.com/projects/BeautyGAN)

논문과 데이터셋은 제공되지만, 소스 코드와 훈련된 모델은 제공되지 않습니다.

### 재현 결과

![](result.jpg)

### 사용 방법

- Python3.6
- TensorFlow1.9

훈련된 모델 다운로드

- [https://pan.baidu.com/s/1wngvgT0qzcKJ5LfLMO7m8A](https://pan.baidu.com/s/1wngvgT0qzcKJ5LfLMO7m8A), 비밀번호: 7lip
- [https://drive.google.com/drive/folders/1pgVqnF2-rnOxcUQ3SO4JwHUFTdiSe5t9](https://drive.google.com/drive/folders/1pgVqnF2-rnOxcUQ3SO4JwHUFTdiSe5t9)

`model`이라는 새 폴더를 만들고, 모델 파일을 그 안에 넣습니다.

`imgs` 폴더에는 11장의 무화장 사진과 9장의 메이크업 사진이 포함되어 있습니다.

기본적으로 `imgs/no_makeup/xfsy_0068.png`에 메이크업을 적용합니다.

