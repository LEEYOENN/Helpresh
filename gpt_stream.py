import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image, ImageOps

# =========================
# 스타일 팔레트 (화이트 + 라이트 그린)
# =========================
PRIMARY = "#34D399"     # 밝은 연두(에메랄드)
PRIMARY_SOFT = "#D1FAE5" # 아주 옅은 연두 배경
BORDER = "#A7F3D0"       # 라이트 그린 보더
TEXT_DARK = "#1F2937"    # 제목/진한 텍스트
TEXT_SOFT = "#6B7280"    # 보조 텍스트
CARD_BG = "#FFFFFF"      # 카드(화이트)
BG_GRAD_TOP = "#FFFFFF"  # 배경 그라데이션 위
BG_GRAD_BOTTOM = "#F5FFFA"  # 배경 그라데이션 아래(민트 기운)

# 예측할 클래스들
class_names = ['금속캔알루미늄캔', '금속캔철캔', '비닐', '스티로폼','유리병갈색', 
               '유리병녹색', '유리병투명', '종이', '페트병무색단일', '페트병유색단일', '플라스틱PE', '플라스틱PP', '플라스틱PS']

# =========================
# 페이지 기본 설정 & 글로벌 CSS
# =========================
st.set_page_config(page_title="웹캠 이미지 분류기", page_icon="📸", layout="centered")

st.markdown(f"""
<style>
/* 전체 배경 그라데이션 */
.stApp {{
    background: linear-gradient(180deg, {BG_GRAD_TOP} 0%, {BG_GRAD_BOTTOM} 100%);
}}

/* 메인 컨테이너 폭 & 패딩 */
.main .block-container {{
    padding-top: 1.2rem;
    max-width: 980px;
}}

/* 공통 카드 스타일 */
.card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 6px 24px rgba(16, 185, 129, 0.08);
}}

/* 헤더 카드 */
.header-card h1 {{
    color: {TEXT_DARK};
    margin: 0 0 10px 0;
    font-size: 2.0rem;
    letter-spacing: -0.3px;
}}
.header-card p {{
    color: {TEXT_SOFT};
    margin: 0;
    font-size: 1.05rem;
}}

/* 결과 카드 */
.result-title {{
    color: {TEXT_DARK};
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 8px;
}}
.predicted-chip {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: {PRIMARY_SOFT};
    color: {TEXT_DARK};
    border: 1px solid {BORDER};
    padding: 10px 14px;
    border-radius: 999px;
    font-weight: 600;
    margin-top: 4px;
}}
.confidence-pill {{
    display: inline-block;
    font-weight: 700;
    color: white;
    background: {PRIMARY};
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 0.9rem;
}}

/* 진행바(확률 바) */
.prog {{
    width: 100%;
    height: 14px;
    background: #ECFDF5;
    border: 1px solid {BORDER};
    border-radius: 999px;
    overflow: hidden;
}}
.prog > span {{
    display: block;
    height: 100%;
    background: {PRIMARY};
    transition: width 400ms ease;
}}

/* 카메라 박스 테두리 느낌 */
[data-testid="stCamera"] > div {{
    border: 1px dashed {BORDER};
    border-radius: 16px !important;
}}

/* 버튼 스타일 */
div.stButton > button:first-child {{
    background: {PRIMARY};
    color: white;
    border: 0;
    border-radius: 12px;
    padding: 0.6rem 1.0rem;
    font-weight: 700;
}}
div.stButton > button:hover {{
    filter: brightness(1.02);
    transform: translateY(-1px);
}}
</style>
""", unsafe_allow_html=True)

# =========================
# 모델 로드
# =========================
@st.cache_resource
def load_model():
    st.info("모델 로딩 중...")
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=len(class_names), bias=True)

    # ⚠️ 실제 환경에 맞는 가중치 경로로 교체하세요.
    # 예: "model/best_efficient_model_epoch12_weights.pth"
    state = torch.load("model/best_efficient_model_epoch{epoch}_weights.pth", map_location="cpu")
    model.load_state_dict(state)

    model.eval()
    st.success("모델 로드 완료!")
    return model

# =========================
# 전처리
# =========================
def transform_image(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transforms_test(image).unsqueeze(0)

# =========================
# 헤더
# =========================
st.markdown(f"""
<div class="card header-card" style="background:{PRIMARY_SOFT}; border-color:{BORDER}">
  <h1>📸 웹캠 이미지 분류기</h1>
  <p>웹캠으로 촬영 → 모델 예측 → 결과를 시각적으로 확인하세요.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# 레이아웃: 좌(카메라) / 우(결과)
# =========================
left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1) 사진 촬영")
    st.caption("촬영 후 ‘분석 시작’ 버튼을 눌러 주세요.")
    cam_image = st.camera_input("웹캠")
    analyze = st.button("분석 시작")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2) 예측 결과")

    if analyze and cam_image is not None:
        with st.spinner("이미지 예측 중..."):
            try:
                pil_image = Image.open(cam_image)
                x = transform_image(pil_image)
                model = load_model()

                with torch.no_grad():
                    logits = model(x)
                    probs = F.softmax(logits, dim=1).squeeze(0)  # [C]

                # Top-1
                top1_prob, top1_idx = torch.topk(probs, 1)
                predicted = class_names[int(top1_idx)]
                conf = float(top1_prob.item()) * 100

                # Top-3 표시용
                topk = 3 if len(class_names) >= 3 else len(class_names)
                top_probs, top_indices = torch.topk(probs, topk)
                top_items = [(class_names[int(i)], float(p.item())*100) for p, i in zip(top_probs, top_indices)]

                # 예측 카드 뱃지
                st.markdown(f"""
                    <div class="result-title">✨ 예측 결과</div>
                    <div class="predicted-chip">
                        <span>예측 클래스:</span>
                        <span style="font-weight:800">{predicted}</span>
                        <span class="confidence-pill">{conf:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)

                st.write("")
                st.markdown("**상위 예측 확률**")
                # 진행바 형태로 Top-3 확률 시각화
                for label, pct in top_items:
                    pct_clamped = max(0.0, min(100.0, pct))
                    st.markdown(f"""
                        <div style="margin: 10px 0 14px 0;">
                            <div style="display:flex; justify-content:space-between; font-size:0.95rem;">
                                <span style="color:{TEXT_DARK}; font-weight:600">{label}</span>
                                <span style="color:{TEXT_SOFT};">{pct_clamped:.2f}%</span>
                            </div>
                            <div class="prog"><span style="width:{pct_clamped}%;"></span></div>
                        </div>
                    """, unsafe_allow_html=True)

                # 원본 프리뷰
                with st.expander("원본 이미지 미리보기", expanded=False):
                    st.image(pil_image, use_container_width=True)

                # 신뢰도 높으면 축하 애니메이션!
                if conf >= 90:
                    st.balloons()

            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {e}")
    else:
        st.info("촬영 후 **분석 시작**을 눌러 예측을 진행하세요.")
        st.markdown("""
        <ul style="color:#4B5563; margin-top:6px;">
          <li>배경을 단순하게, 사물을 화면 중앙에 두면 더 잘 맞춥니다.</li>
          <li>빛 반사가 심하면 각도를 조금 바꿔 촬영해 보세요.</li>
        </ul>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 하단 안내/풋터
# =========================
st.write("")
with st.expander("ℹ️ 참고 정보 / 사용법"):
    st.markdown(f"""
- 이 앱은 **PyTorch + Streamlit** 기반입니다.  
- 전처리는 EfficientNet 계열의 일반적인 정규화를 사용했습니다.  
- 가중치 파일 경로를 실제 파일로 교체하세요.  
  - 예: `model/best_efficient_model_epoch12_weights.pth`  
- 결과는 입력 이미지 품질과 학습 데이터 분포에 따라 달라질 수 있습니다.
    """)

st.markdown(
    f"""
    <div style="text-align:center; color:{TEXT_SOFT}; font-size:0.9rem; margin-top:12px;">
        Made with LEE YEON using Streamlit · Theme: White + Light Green
    </div>
    """,
    unsafe_allow_html=True
)
