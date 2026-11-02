import pandas as pd
import plotly.express as px
import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="데이터 탐색 - 뇌졸중 예측 실습실", page_icon="📊", layout="wide")

# 데이터 로드 (캐싱 처리 및 UTF-8 인코딩)
DATA_URL = "https://raw.githubusercontent.com/20218sonyongjun/data/refs/heads/main/stroke.csv"


@st.cache_data
def load_data():
  return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()

st.title("📊 뇌졸중 데이터 탐색 페이지")
st.write(
    "stroke.csv 데이터를 활용하여 주요 변수들의 분포와 뇌졸중 간의 관계를"
    " 시각화하고 탐색합니다."
)
st.markdown("---")

# -----------------------------------------------------------------------------
# 1. 나이와 평균 혈당의 분포 (히스토그램 2개 나란히)
# -----------------------------------------------------------------------------
st.subheader("1. 나이(Age) 및 평균 혈당(Avg Glucose Level) 분포")

col1, col2 = st.columns(2)

with col1:
  fig_age = px.histogram(
      df,
      x="age",
      title="나이(Age) 분포",
      labels={"age": "나이"},
      color_discrete_sequence=["skyblue"],
  )
  st.plotly_chart(fig_age, use_container_width=True)

with col2:
  fig_glucose = px.histogram(
      df,
      x="avg_glucose_level",
      title="평균 혈당(Avg Glucose Level) 분포",
      labels={"avg_glucose_level": "평균 혈당"},
      color_discrete_sequence=["salmon"],
  )
  st.plotly_chart(fig_glucose, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교 (상자그림 & 평균값 표)
# -----------------------------------------------------------------------------
st.subheader("2. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교")

df_plot = df.copy()
df_plot["stroke_label"] = df_plot["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})

col1, col2 = st.columns(2)

with col1:
  fig_box_age = px.box(
      df_plot,
      x="stroke_label",
      y="age",
      color="stroke_label",
      title="뇌졸중 여부에 따른 나이 상자그림",
      labels={"stroke_label": "뇌졸중 여부", "age": "나이"},
  )
  st.plotly_chart(fig_box_age, use_container_width=True)

with col2:
  fig_box_glucose = px.box(
      df_plot,
      x="stroke_label",
      y="avg_glucose_level",
      color="stroke_label",
      title="뇌졸중 여부에 따른 평균 혈당 상자그림",
      labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"},
  )
  st.plotly_chart(fig_box_glucose, use_container_width=True)

# 두 그룹의 평균값 표
st.write("#### 📋 그룹별 평균값 요약 표")
mean_df = df.groupby("stroke")[["age", "avg_glucose_level"]].mean().reset_index()
mean_df["stroke"] = mean_df["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})
mean_df.columns = ["뇌졸중 여부", "평균 나이", "평균 혈당"]
st.dataframe(mean_df, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 3. 고혈압 및 심장병 유무에 따른 뇌졸중 비율 (막대그래프)
# -----------------------------------------------------------------------------
st.subheader("3. 고혈압 및 심장병 유무에 따른 뇌졸중 비율")

col1, col2 = st.columns(2)

with col1:
  hyp_stroke = df.groupby("hypertension")["stroke"].mean().reset_index()
  hyp_stroke["hypertension"] = hyp_stroke["hypertension"].map(
      {0: "고혈압 없음", 1: "고혈압 있음"}
  )
  hyp_stroke["stroke_rate"] = hyp_stroke["stroke"] * 100

  fig_hyp = px.bar(
      hyp_stroke,
      x="hypertension",
      y="stroke_rate",
      text="stroke_rate",
      title="고혈압 유무에 따른 뇌졸중 비율 (%)",
      labels={"hypertension": "고혈압 여부", "stroke_rate": "뇌졸중 비율 (%)"},
      color="hypertension",
  )
  fig_hyp.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
  st.plotly_chart(fig_hyp, use_container_width=True)

with col2:
  heart_stroke = df.groupby("heart_disease")["stroke"].mean().reset_index()
  heart_stroke["heart_disease"] = heart_stroke["heart_disease"].map(
      {0: "심장병 없음", 1: "심장병 있음"}
  )
  heart_stroke["stroke_rate"] = heart_stroke["stroke"] * 100

  fig_heart = px.bar(
      heart_stroke,
      x="heart_disease",
      y="stroke_rate",
      text="stroke_rate",
      title="심장병 유무에 따른 뇌졸중 비율 (%)",
      labels={"heart_disease": "심장병 여부", "stroke_rate": "뇌졸중 비율 (%)"},
      color="heart_disease",
  )
  fig_heart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
  st.plotly_chart(fig_heart, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. BMI 결측치 그룹과 전체 뇌졸중 비율 비교 표
# -----------------------------------------------------------------------------
st.subheader("4. 체질량지수(BMI) 결측치 그룹 vs 전체 뇌졸중 비율 비교")

total_count = len(df)
total_stroke_rate = (df["stroke"].sum() / total_count) * 100

bmi_null_df = df[df["bmi"].isnull()]
null_count = len(bmi_null_df)
null_stroke_rate = (
    (bmi_null_df["stroke"].sum() / null_count * 100) if null_count > 0 else 0.0
)

comparison_data = [
    {
        "구분": "전체 데이터",
        "인원수": total_count,
        "뇌졸중 환자수": int(df["stroke"].sum()),
        "뇌졸중 비율 (%)": round(total_stroke_rate, 2),
    },
    {
        "구분": "BMI 비어 있는 사람 (결측치)",
        "인원수": null_count,
        "뇌졸중 환자수": int(bmi_null_df["stroke"].sum()),
        "뇌졸중 비율 (%)": round(null_stroke_rate, 2),
    },
]

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. 흡연 상태(smoking_status)별 인원수 표
# -----------------------------------------------------------------------------
st.subheader("5. 흡연 상태(smoking_status)별 인원수")

smoking_counts = (
    df["smoking_status"]
    .value_counts()
    .reset_index(name="인원수")
    .rename(columns={"smoking_status": "흡연 상태"})
)
st.dataframe(smoking_counts, use_container_width=True)
