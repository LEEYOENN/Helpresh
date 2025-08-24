import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

# 예측할 13개 클래스의 이름을 정의합니다.
# 실제 모델이 학습한 클래스 이름으로 교체해야 합니다.
class_names = ['금속캔알루미늄캔', '금속캔철캔', '비닐', '스티로폼','유리병갈색', 
               '유리병녹색', '유리병투명', '종이', '페트병무색단일', '페트병유색단일', '플라스틱PE', '플라스틱PP', '플라스틱PS']

@st.cache_resource
def load_model():
    """
    미리 학습된 모델을 로드하고 캐시합니다.
    여기서는 EfficientNetB0 모델 구조를 시뮬레이션하고,
    사용자가 요청한 대로 분류기(classifier)를 수정했습니다.
    """
    st.info("모델 로딩 중...")
    
    # EfficientNetB0 모델 구조를 불러옵니다.
    # weights=None으로 지정하여 가중치를 불러오지 않습니다.
    # 대신, 사용자가 지정한 모델 구조로 classifier를 대체합니다.
    model = efficientnet_b0(weights=None)
    
    # 사용자가 요청한 대로 모델의 분류기 레이어를 수정합니다.
    model.classifier[1] = nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
    
    
    # 여기서는 실제 .pth 파일을 로드하는 대신,
    # 더미 상태 딕셔너리를 생성하여 모델을 로드한 것처럼 만듭니다.
    # 실제 환경에서는 아래 두 줄을 모델 파일을 로드하는 코드로 교체해야 합니다.
    # st.info("더미 가중치를 로드합니다. 실제 모델 가중치로 교체하세요.")
    dummy_state_dict = torch.load("model/best_efficient_model_epoch{epoch}_weights.pth",
                                  map_location=torch.device('cpu'))
    model.load_state_dict(dummy_state_dict)

    # 모델을 평가 모드로 설정합니다.
    model.eval()
    
    st.success("모델 로드 완료!")
    return model

def transform_image(image: Image.Image):
    """
    PIL 이미지를 모델 입력에 맞게 변환합니다.
    """
    # EfficientNetB0_Weights.DEFAULT.transforms()를 사용하여
    # 사전 학습된 모델에 맞는 표준 전처리 파이프라인을 사용합니다.
    # preprocess = EfficientNet_B0_Weights.DEFAULT.transforms()

    transforms_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transforms_test(image).unsqueeze(0)

# --- Streamlit 애플리케이션 UI 구성 ---
st.set_page_config(page_title="웹캠 이미지 분류기", layout="centered")

st.markdown(
    """
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #4CAF50; font-family: 'Arial Black', sans-serif;">
            📸 웹캠 이미지 분류기
        </h1>
        <p style="font-size: 1.2em; color: #555;">
            아래 웹캠으로 사진을 찍어보세요! 모델이 이미지를 예측해 드립니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# 웹캠 입력 위젯
cam_image = st.camera_input("웹캠")

if cam_image is not None:
    # 이미지가 캡처되면 이 블록이 실행됩니다.
    
    with st.spinner('이미지 예측 중...'):
        try:
            # BytesIO를 통해 PIL 이미지로 변환
            pil_image = Image.open(cam_image)
            
            # 모델 입력에 맞게 이미지 전처리
            transformed_image = transform_image(pil_image)
            
            # 모델 로드
            model = load_model()
            
            # 예측 수행
            with torch.no_grad():
                logits = model(transformed_image)
                probabilities = F.softmax(logits, dim=1)
                
                # 가장 높은 확률을 가진 클래스 찾기
                top_prob, top_class = torch.topk(probabilities, 1)
                
                # 예측 결과 추출
                predicted_class = top_class.item()
                confidence = top_prob.item() * 100
            
            # 예측 결과를 예쁘게 표시
            st.markdown(
                f"""
                <div style="text-align: center; padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f9f9f9;">
                    <h3 style="color: #333;">✨ 예측 결과</h3>
                    <p style="font-size: 1.5em; font-weight: bold; color: #4CAF50;">
                        "{class_names[predicted_class]}"
                    </p>
                    <p style="font-size: 1.2em; color: #666;">
                        확신도: {confidence:.2f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")

st.write("---")
st.markdown(
    """
    <div style="text-align: center; padding: 10px;">
        <p style="font-size: 0.9em; color: #999;">
            이 애플리케이션은 PyTorch와 Streamlit으로 제작되었습니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
