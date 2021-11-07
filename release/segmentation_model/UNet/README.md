# UNet 계열의 모델 실험

UNet을 기반으로한 모델들을 실험할 수 있는 코드가 작성된 디렉토리입니다.  
아래와 같은 모델들의 훈련이 작성되어 있습니다.  
- UNet
- UNet2+
- UNet3+
- UNet3+ with Deep Supervision
- UNet3+ with Deep Supervision and CGM
- ResUNet3+ with Deep Supervision and CGM
- MMSegmentaion UNet

## UNet-mmseg
UNet-mmseg 에 위치한 코드들을 통해서 mmsegmentation의 다양한 backbone을 가진 UNet을 훈련할 수 있습니다. 동일한 pip 환경은 `pip install -r ./UNet-mmseg/mmsegUNet.txt`를 실행하여 설치가 가능합니다. mmsegmentation v0.19.0 을 기준으로 작성되었으며 `https://github.com/open-mmlab/mmsegmentation`을 통해 수동으로 설치가 가능합니다. 

### 실험방법
1. mmsegmentation과 다른 필요 라이브러리를 설치합니다.
2. 실험 템플릿인 exp_00 을 복사하여 원하는 이름으로 변경합니다.
3. `mm_config`로 원하는 config 파일의 필요 부분을 모두 복사합니다.
4. `python mm_config/merge_config.py` 를 실행하여 같은 디렉토리에 `default_cofnig.py` 를 생성합니다.
5. `default_cofnig.py` 를 수정하여 원하는 config file을 작성합니다.
6. `sh run_train.sh` 를 실행하여 학습을 진행합니다.

## UNet3+

여러 버전의 UNet을 미리 학습된 파라미터 없이 처음부터 학습할 수 있습니다. `pip install -r UNet3p.txt` 를 이용해 환경을 맞춰줍니다. 그리고 `UNet3+/UNet/models/` 에 위치한 모델들을 import하여 `python UNet3+/UNet/UNet3p.py/` 를 통해서 학습이 가능합니다. ResUNet 을 제외한 다른 코드들의 출처는 `https://github.com/ZJUGiveLab/UNet-Version` 이며 ResUNet3+의 경우 pytorch의 pretrained ResNet101 을 encoder로 하는 UNet3+ 구조의 모델입니다.
