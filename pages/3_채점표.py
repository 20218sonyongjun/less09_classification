import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="채점표 - 뇌졸중 예측 실습실", page_icon="📋", layout="wide"
)

# 데이터 로드
DATA_URL = "https://raw.githubusercontent.com/20218sonyongjun/data/refs/heads/main/stroke.csv"


@st.cache_data
def load_data():
  return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()

st.title("📋 모델 채점표 및 오류 분석")
st.write(
    "테스트 데이터를 바탕으로 로지스틱 회귀와 의사결정트리 모델의 혼동행렬,"
    " 성능 지표, 그리고 예측 오류 사례를 분석합니다."
)
st.markdown("---")

# -----------------------------------------------------------------------------
# 1. 입력 속성(Features) 선택 (분류 모델 페이지와 동일한 구성)
# -----------------------------------------------------------------------------
st.subheader("1. 입력 속성(Features) 선택")

feature_dict = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

all_features = list(feature_dict.keys())
default_features = [f for f in all_features if f != "bmi"]

selected_kr_names = st.multiselect(
    "모델 학습에 사용할 속성을 선택하세요 (2개 이상)",
    options=list(feature_dict.values()),
    default=[feature_dict[f] for f in default_features],
    key="score_features",
)

kr_to_feature = {v: k for k, v in feature_dict.items()}
selected_features = [kr_to_feature[name] for name in selected_kr_names]

if len(selected_features) < 2:
  st.warning(
      "⚠️ 속성을 **2개 이상** 선택해 주세요. (현재 선택된 속성 수:"
      f" {len(selected_features)}개)"
  )
  st.stop()

st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 데이터 전처리 및 번호 순 분할 (Train / Test Split)
# -----------------------------------------------------------------------------
if "id" in df.columns:
  df_sorted = df.sort_values("id").reset_index(drop=True)
else:
  df_sorted = df.reset_index(drop=True)

test_indices = []
train_indices = []
n_rows = len(df_sorted)

for i in range(0, n_rows, 10):
  chunk = list(range(i, min(i + 10, n_rows)))
  if len(chunk) == 10:
    test_indices.extend(chunk[:3])
    train_indices.extend(chunk[3:])
  else:
    split_pt = int(len(chunk) * 0.3)
    test_indices.extend(chunk[:split_pt])
    train_indices.extend(chunk[split_pt:])

df_train = df_sorted.iloc[train_indices].copy()
df_test = df_sorted.iloc[test_indices].copy()

if "bmi" in selected_features:
  bmi_median = df_train["bmi"].median()
  df_train["bmi"] = df_train["bmi"].fillna(bmi_median)
  df_test["bmi"] = df_test["bmi"].fillna(bmi_median)

X_train = df_train[selected_features]
y_train = df_train["stroke"]
X_test = df_test[selected_features]
y_test = df_test["stroke"]

# -----------------------------------------------------------------------------
# 3. 모델 학습 및 예측
# -----------------------------------------------------------------------------
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

dt_model = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_model.fit(X_train, y_train)
y_pred_dummy = dummy_model.predict(X_test)

# 테스트 데이터 요약 정보
total_test_count = len(y_test)
actual_stroke_count = int(y_test.sum())
st.markdown(
    f"📌 **테스트용 사람 수:** {total_test_count:,}명 | **그중 실제 뇌졸중인 사람"
    f" 수:** {actual_stroke_count:,}명"
)
st.markdown("---")


# 혼동행렬 계산 함수
def calc_confusion_matrix(y_true, y_pred):
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  tn = int(np.sum((y_true == 0) & (y_pred == 0)))
  fp = int(np.sum((y_true == 0) & (y_pred == 1)))
  fn = int(np.sum((y_true == 1) & (y_pred == 0)))
  tp = int(np.sum((y_true == 1) & (y_pred == 1)))
  return tn, fp, fn, tp


cm_lr = calc_confusion_matrix(y_test, y_pred_lr)
cm_dt = calc_confusion_matrix(y_test, y_pred_dt)
cm_dummy = calc_confusion_matrix(y_test, y_pred_dummy)


# 지표 계산 함수
def get_metrics(cm):
  tn, fp, fn, tp = cm
  total = tn + fp + fn + tp
  acc = (tp + tn) / total if total > 0 else 0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  precision = tp / (tp + fp) if (tp + fp) > 0 else None
  if precision is not None and (precision + recall) > 0:
    f1 = (2 * precision * recall) / (precision + recall)
  else:
    f1 = None
  return acc, recall, precision, f1


metrics_lr = get_metrics(cm_lr)
metrics_dt = get_metrics(cm_dt)
metrics_dummy = get_metrics(cm_dummy)

# -----------------------------------------------------------------------------
# 4. 혼동행렬 (2×2 표) 시각화
# -----------------------------------------------------------------------------
st.subheader("2. 모델별 혼동행렬 (Confusion Matrix)")
st.write(
    "행은 **실제 값**, 열은 **예측 값**을 나타내며, 각 칸에는 사람 수와 기호(TN,"
    " FP, FN, TP)가 함께 표시됩니다."
)

col1, col2 = st.columns(2)


def make_cm_dataframe(cm):
  tn, fp, fn, tp = cm
  data = {
      "예측: 정상(0)": [f"TN: {tn:,}명", f"FN: {fn:,}명", f"합계: {tn+fn:,}명"],
      "예측: 뇌졸중(1)": [f"FP: {fp:,}명", f"TP: {tp:,}명", f"합계: {fp+tp:,}명"],
      "합계": [
          f"합계: {tn+fp:,}명",
          f"합계: {fn+tp:,}명",
          f"전체: {tn+fp+fn+tp:,}명",
      ],
  }
  return pd.DataFrame(
      data, index=["실제: 정상(0)", "실제: 뇌졸중(1)", "합계"]
  )


with col1:
  st.markdown("#### 로지스틱 회귀(확률로 답하는 모델)")
  st.dataframe(make_cm_dataframe(cm_lr), use_container_width=True)

with col2:
  st.markdown("#### 의사결정트리(질문으로 답하는 모델)")
  st.dataframe(make_cm_dataframe(cm_dt), use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. 성능 지표 비교 표
# -----------------------------------------------------------------------------
st.subheader("3. 모델 성능 지표 비교")


def format_metric_row(name, metrics):
  acc, recall, prec, f1 = metrics
  prec_str = f"{prec*100:.2f}%" if prec is not None else "계산할 수 없음"
  f1_str = f"{f1:.4f}" if f1 is not None else "계산할 수 없음"
  return {
      "모델": name,
      "정확도 (Accuracy)": f"{acc*100:.2f}%",
      "재현율 (Recall)": f"{recall*100:.2f}%",
      "정밀도 (Precision)": prec_str,
      "F1 점수": f1_str,
  }


table_data = [
    format_metric_row("로지스틱 회귀(확률로 답하는 모델)", metrics_lr),
    format_metric_row("의사결정트리(질문으로 답하는 모델)", metrics_dt),
    format_metric_row(
        "훈련용에서 많은 쪽으로만 답하는 모델 (다수결 모델)", metrics_dummy
    ),
]

metrics_df = pd.DataFrame(table_data)
st.dataframe(metrics_df, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. 오류 분석 목록 (놓친 환자 & 헛짚은 사람)
# -----------------------------------------------------------------------------
st.subheader("4. 예측 오류 상세 분석 (놓친 환자 vs 헛짚은 사람)")

model_choice_name = st.radio(
    "분석할 모델을 선택하세요",
    [
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
    ],
    horizontal=True,
)

if model_choice_name == "로지스틱 회귀(확률로 답하는 모델)":
  chosen_pred = y_pred_lr
  chosen_cm = cm_lr
  current_model_label = "로지스틱 회귀"
else:
  chosen_pred = y_pred_dt
  chosen_cm = cm_dt
  current_model_label = "의사결정트리"

# 데이터프레임 구성
df_error_analysis = df_test.copy()
df_error_analysis["predicted"] = chosen_pred

# 왼쪽: 실제 뇌졸중인데 아니라고 예측한 사람 (놓친 환자 / FN)
fn_df = df_error_analysis[
    (df_error_analysis["stroke"] == 1)
    & (df_error_analysis["predicted"] == 0)
].copy()

# 오른쪽: 아니는데 뇌졸중이라고 예측한 사람 (헛짚은 사람 / FP)
fp_df = df_error_analysis[
    (df_error_analysis["stroke"] == 0)
    & (df_error_analysis["predicted"] == 1)
].copy()

# 나이가 많은 순서로 정렬
fn_df = fn_df.sort_values("age", ascending=False)
fp_df = fp_df.sort_values("age", ascending=False)

total_fn = len(fn_df)
total_fp = len(fp_df)

# 포맷팅 함수
def format_error_table(subset_df):
  if subset_df.empty:
    return pd.DataFrame(
        columns=["번호", "나이", "평균 혈당", "고혈압 여부", "심장병 여부"]
    )

  formatted_rows = []
  for idx, row in subset_df.iterrows():
    # 번호 (id가 있으면 id 사용, 없으면 인덱스)
    row_id = row["id"] if "id" in row and pd.notna(row["id"]) else idx
    hyp_str = "고혈압 있음" if row["hypertension"] == 1 else "고혈압 없음"
    heart_str = "심장병 있음" if row["heart_disease"] == 1 else "심장병 없음"

    formatted_rows.append({
        "번호": row_id,
        "나이": row["age"],
        "평균 혈당": row["avg_glucose_level"],
        "고혈압 여부": hyp_str,
        "심장병 여부": heart_str,
    })
  return pd.DataFrame(formatted_rows)


fn_formatted = format_error_table(fn_df.head(20))
fp_formatted = format_error_table(fp_df.head(20))

col1, col2 = st.columns(2)

with col1:
  title_fn = (
      f"🔴 놓친 환자 목록 (실제 뇌졸중인데 아님 예측) - 전체 {total_fn}명"
      + (" (상위 20명 표시)" if total_fn > 20 else "")
  )
  st.markdown(f"#### {title_fn}")
  if not fn_formatted.empty:
    st.dataframe(fn_formatted, use_container_width=True)
  else:
    st.info("해당하는 오류 환자가 없습니다.")

with col2:
  title_fp = (
      f"🔵 헛짚은 사람 목록 (실제 아닌데 뇌졸중 예측) - 전체 {total_fp}명"
      + (" (상위 20명 표시)" if total_fp > 20 else "")
  )
  st.markdown(f"#### {title_fp}")
  if not fp_formatted.empty:
    st.dataframe(fp_formatted, use_container_width=True)
  else:
    st.info("해당하는 오류 인원이 없습니다.")

st.markdown("---")
st.success(
    f"💡 **검증 확인:** 선택하신 **{current_model_label}** 모델의 혼동행렬상"
    f" **FN(놓친 환자) 칸의 수({chosen_cm[2]}명)**는 위 왼쪽 목록의 전체"
    f" 인원수({total_fn}명)와 정확히 일치하며, **FP(헛짚은 사람)"
    f" 칸의 수({chosen_cm[1]}명)**는 위 오른쪽 목록의 전체"
    f" 인원수({total_fp}명)와 정확히 일치합니다."
)
