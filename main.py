import json
import os
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. 페이지 설정 (아이콘 및 브라우저 탭 제목)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. JSON 데이터 저장/불러오기 기능 (내_기록/data.json)
# -----------------------------------------------------------------------------
SAVE_DIR = "내_기록"
SAVE_FILE = os.path.join(SAVE_DIR, "data.json")

def load_user_notes():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"meanings": {}, "source": ""}

def save_user_notes(notes):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

user_notes = load_user_notes()

# -----------------------------------------------------------------------------
# 3. 데이터 로드 (캐싱 처리)
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/20218sonyongjun/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")

df = load_data()

# -----------------------------------------------------------------------------
# 4. 앱 헤더 및 핵심 지표 카드 (4개)
# -----------------------------------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.write("stroke.csv 데이터를 활용하여 뇌졸중 관련 요인을 탐색하고 분석하는 실습 공간입니다.")
st.markdown("---")

total_people = len(df)
total_cols = len(df.columns)
stroke_count = int(df["stroke"].sum())
stroke_ratio = (stroke_count / total_people) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric(label="전체 사람 수", value=f"{total_people:,} 명")
col2.metric(label="열 개수", value=f"{total_cols} 개")
col3.metric(label="stroke = 1 (환자 수)", value=f"{stroke_count:,} 명")
col4.metric(label="뇌졸중 비율", value=f"{stroke_ratio:.2f} %")

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. 열 정보 및 우리말 뜻 작성 표
# -----------------------------------------------------------------------------
st.subheader("📋 데이터 열 정보 및 우리말 뜻 정리")
st.caption("교재를 참고하여 각 열의 우리말 뜻을 입력하고 [기록 저장하기] 버튼을 눌러주세요.")

default_meanings = user_notes.get("meanings", {})

info_data = []
for col in df.columns:
    null_cnt = int(df[col].isnull().sum())
    dtype_str = str(df[col].dtype)
    info_data.append({
        "열 이름": col,
        "우리말 뜻": default_meanings.get(col, ""),
        "값의 종류": dtype_str,
        "빈 값 개수": null_cnt
    })

info_df = pd.DataFrame(info_data)

edited_df = st.data_editor(
    info_df,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    use_container_width=True,
    key="meaning_editor"
)

# -----------------------------------------------------------------------------
# 6. 데이터 출처 작성
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📌 데이터 출처")

source_text = st.text_area(
    "교재에 기재된 데이터 출처 정보를 적어주세요.",
    value=user_notes.get("source", ""),
    height=100
)

if st.button("💾 기록 저장하기", type="primary"):
    updated_meanings = {
        row["열 이름"]: row["우리말 뜻"]
        for _, row in edited_df.iterrows()
    }
    new_notes = {
        "meanings": updated_meanings,
        "source": source_text
    }
    save_user_notes(new_notes)
    st.success("내_기록/data.json 파일에 정상적으로 저장되었습니다!")

# -----------------------------------------------------------------------------
# 7. 데이터 상위 5개 행 출력
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 데이터 미리보기 (처음 5개 행)")
st.dataframe(df.head(5), use_container_width=True)
