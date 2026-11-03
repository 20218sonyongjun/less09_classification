import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="성능 높이기 - 뇌졸중 예측 실습실", page_icon="⚖️", layout="wide"
)

# 데이터 로드
DATA_URL = "https://raw.githubusercontent.com/20218sonyongjun/data/refs/heads/main/stroke.csv"


@st.cache_data
def load_data():
  return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()

st.title("⚖️ 클래스 불균형 해결 및 모델 성능 높이기")
st.write(
    "뇌졸중 데이터는 정상 환자에 비해 뇌졸중 환자가 매우 드문(불균형) 데이터입니다."
    " 가중치 조절(class_weight='balanced') 전후의 성능 변화와 결정 경계를"
    " 비교합니다."
)
st.markdown("---")

# -----------------------------------------------------------------------------
# 1. 입력 속성(Features) 선택
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
# 처음에는 age, avg_glucose_level, hypertension, heart_disease 네 가지가 골라져 있게 설정 (bmi 제외)
default_features = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

selected_kr_names = st.multiselect(
    "모델 학습에 사용할 속성을 선택하세요 (2개 이상)",
    options=list(feature_dict.values()),
    default=[feature_dict[f] for f in default_features],
    key="boost_features",
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

# 체질량지수(bmi)는 비어 있는 사람이 있으니 고른 경우에만 훈련용의 중앙값으로 채우기
if "bmi" in selected_features:
  bmi_median = df_train["bmi"].median()
  df_train["bmi"] = df_train["bmi"].fillna(bmi_median)
  df_test["bmi"] = df_test["bmi"].fillna(bmi_median)

X_train = df_train[selected_features]
y_train = df_train["stroke"]
X_test = df_test[selected_features]
y_test = df_test["stroke"]

# -----------------------------------------------------------------------------
# 3. 모델 학습 (가중치 적용 전후)
# -----------------------------------------------------------------------------
# 1) 로지스틱 회귀
lr_before = LogisticRegression(random_state=42, max_iter=1000)
lr_before.fit(X_train, y_train)
y_pred_lr_before = lr_before.predict(X_test)

lr_after = LogisticRegression(
    class_weight="balanced", random_state=42, max_iter=1000
)
lr_after.fit(X_train, y_train)
y_pred_lr_after = lr_after.predict(X_test)

# 2) 의사결정트리
dt_before = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42
)
dt_before.fit(X_train, y_train)
y_pred_dt_before = dt_before.predict(X_test)

dt_after = DecisionTreeClassifier(
    class_weight="balanced", max_depth=3, min_samples_leaf=5, random_state=42
)
dt_after.fit(X_train, y_train)
y_pred_dt_after = dt_after.predict(X_test)


# 혼동행렬 및 지표 계산 함수
def calc_cm(y_true, y_pred):
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  tn = int(np.sum((y_true == 0) & (y_pred == 0)))
  fp = int(np.sum((y_true == 0) & (y_pred == 1)))
  fn = int(np.sum((y_true == 1) & (y_pred == 0)))
  tp = int(np.sum((y_true == 1) & (y_pred == 1)))
  return tn, fp, fn, tp


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


cm_lr_b = calc_cm(y_test, y_pred_lr_before)
cm_lr_a = calc_cm(y_test, y_pred_lr_after)
cm_dt_b = calc_cm(y_test, y_pred_dt_before)
cm_dt_a = calc_cm(y_test, y_pred_dt_after)

# -----------------------------------------------------------------------------
# 4. 성능 지표 비교 표 (가중치 전후 나란히)
# -----------------------------------------------------------------------------
st.subheader("2. 가중치 조절 전후 성능 지표 비교")


def make_comparison_row(model_name, cm_b, cm_a):
  acc_b, rec_b, prec_b, f1_b = get_metrics(cm_b)
  acc_a, rec_a, prec_a, f1_a = get_metrics(cm_a)

  prec_b_str = f"{prec_b*100:.2f}%" if prec_b is not None else "계산 불가"
  prec_a_str = f"{prec_a*100:.2f}%" if prec_a is not None else "계산 불가"
  f1_b_str = f"{f1_b:.4f}" if f1_b is not None else "계산 불가"
  f1_a_str = f"{f1_a:.4f}" if f1_a is not None else "계산 불가"

  return {
      "모델": model_name,
      "구분": "가중치 적용 전",
      "정확도": f"{acc_b*100:.2f}%",
      "재현율": f"{rec_b*100:.2f}%",
      "정밀도": prec_b_str,
      "F1 점수": f1_b_str,
  }, {
      "모델": model_name,
      "구분": "가중치 적용 후 (balanced)",
      "정확도": f"{acc_a*100:.2f}%",
      "재현율": f"{rec_a*100:.2f}%",
      "정밀도": prec_a_str,
      "F1 점수": f1_a_str,
  }


row_lr_b, row_lr_a = make_comparison_row(
    "로지스틱 회귀(확률로 답하는 모델)", cm_lr_b, cm_lr_a
)
row_dt_b, row_dt_a = make_comparison_row(
    "의사결정트리(질문으로 답하는 모델)", cm_dt_b, cm_dt_a
)

comp_df = pd.DataFrame([row_lr_b, row_lr_a, row_dt_b, row_dt_a])
st.dataframe(comp_df, use_container_width=True)

# 찾아낸 환자 수(TP), 놓친 환자 수(FN), 헛짚은 사람 수(FP) 변화 요약
st.markdown("#### 📋 환자 탐지 변화 요약 (전 ➔ 후)")
st.markdown(
    f"- **로지스틱 회귀(확률로 답하는 모델)**: 찾아낸 환자 수(TP) {cm_lr_b[3]}명"
    f" ➔ **{cm_lr_a[3]}명** | 놓친 환자 수(FN) {cm_lr_b[2]}명 ➔ **{cm_lr_a[2]}명**"
    f" | 헛짚은 사람 수(FP) {cm_lr_b[1]}명 ➔ **{cm_lr_a[1]}명**"
)
st.markdown(
    f"- **의사결정트리(질문으로 답하는 모델)**: 찾아낸 환자 수(TP)"
    f" {cm_dt_b[3]}명 ➔ **{cm_dt_a[3]}명** | 놓친 환자 수(FN) {cm_dt_b[2]}명 ➔"
    f" **{cm_dt_a[2]}명** | 헛짚은 사람 수(FP) {cm_dt_b[1]}명 ➔ **{cm_dt_a[1]}명**"
)

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. 결정 경계 시각화 (2차원 또는 3차원)
# -----------------------------------------------------------------------------
st.subheader("3. 가중치 조절 전후 0.5 결정 경계 비교 시각화")

# 기타 속성 중앙값 고정용
other_features = [f for f in selected_features if f not in selected_features[:3]]


if len(selected_features) == 2:
  # 2개 선택된 경우 2차원 산점도 및 경계선
  vis_x_col = selected_features[0]
  vis_y_col = selected_features[1]

  st.info(
      "📌 선택된 속성이 2개이므로 **2차원 산점도**와 0.5 결정 경계선을"
      f" 그립니다. (가로축: {feature_dict[vis_x_col]}, 세로축:"
      f" {feature_dict[vis_y_col]})"
  )

  x_min, x_max = df_test[vis_x_col].min(), df_test[vis_x_col].max()
  y_min, y_max = df_test[vis_y_col].min(), df_test[vis_y_col].max()
  x_m = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
  y_m = (y_max - y_min) * 0.1 if y_max > y_min else 1.0

  xx, yy = np.meshgrid(
      np.linspace(x_min - x_m, x_max + x_m, 100),
      np.linspace(y_min - y_m, y_max + y_m, 100),
  )

  grid_dict = {vis_x_col: xx.ravel(), vis_y_col: yy.ravel()}
  for f in other_features:
    grid_dict[f] = np.full(xx.ravel().shape, df_test[f].median())
  X_grid = pd.DataFrame(grid_dict)

  # 로지스틱 회귀 확률 계산
  prob_b = lr_before.predict_proba(X_grid)[:, 1].reshape(xx.shape)
  prob_a = lr_after.predict_proba(X_grid)[:, 1].reshape(xx.shape)

  b_min, b_max = prob_b.min(), prob_b.max()
  a_min, a_max = prob_a.min(), prob_a.max()

  if not ((b_min <= 0.5) and (b_max >= 0.5)):
    st.warning(
        "⚠️ 가중치 적용 전 로지스틱 경계선(0.5)이 현재 그림 범위 밖입니다."
    )
  if not ((a_min <= 0.5) and (a_max >= 0.5)):
    st.warning(
        "⚠️ 가중치 적용 후 로지스틱 경계선(0.5)이 현재 그림 범위 밖입니다."
    )

  fig = go.Figure()

  # 테스트 데이터 산점도
  df_test_vis = df_test.copy()
  df_test_vis["stroke_label"] = df_test_vis["stroke"].map(
      {0: "아님 (0)", 1: "뇌졸중 (1)"}
  )
  color_map = {"아님 (0)": "dodgerblue", "뇌졸중 (1)": "crimson"}

  for label, col in color_map.items():
    subset = df_test_vis[df_test_vis["stroke_label"] == label]
    fig.add_trace(
        go.Scatter(
            x=subset[vis_x_col],
            y=subset[vis_y_col],
            mode="markers",
            name=f"실제: {label}",
            marker=dict(
                color=col, size=8, opacity=0.8, line=dict(width=0.5, color="white")
            ),
        )
    )

  # 가중치 전 경계선 (파란색 점선)
  fig.add_trace(
      go.Contour(
          z=prob_b,
          x=np.linspace(x_min - x_m, x_max + x_m, 100),
          y=np.linspace(y_min - y_m, y_max + y_m, 100),
          contours=dict(
              start=0.5, end=0.5, size=0.1, coloring="lines", showlabels=False
          ),
          line=dict(color="blue", width=2.5, dash="dash"),
          name="가중치 전 경계선 (p=0.5)",
      )
  )

  # 가중치 후 경계선 (주황색 실선)
  fig.add_trace(
      go.Contour(
          z=prob_a,
          x=np.linspace(x_min - x_m, x_max + x_m, 100),
          y=np.linspace(y_min - y_m, y_max + y_m, 100),
          contours=dict(
              start=0.5, end=0.5, size=0.1, coloring="lines", showlabels=False
          ),
          line=dict(color="darkorange", width=2.5),
          name="가중치 후 경계선 (p=0.5)",
      )
  )

  fig.update_layout(
      title=f"로지스틱 회귀 가중치 조절 전후 결정 경계 비교 ({feature_dict[vis_x_col]} vs {feature_dict[vis_y_col]})",
      xaxis_title=feature_dict[vis_x_col],
      yaxis_title=feature_dict[vis_y_col],
      legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
  )
  st.plotly_chart(fig, use_container_width=True)

else:
  # 3개 이상 선택된 경우 3차원 산점도 및 평면 (그중 3개 축 선택 가능)
  st.info(
      "📌 속성이 3개 이상 선택되었으므로 **3차원 산점도**와 전후 0.5"
      " 평면(등치면)을 겹쳐서 보여드립니다. 마우스로 회전해 볼 수 있습니다."
  )

  col_x, col_y, col_z = st.columns(3)
  with col_x:
    axis_x = st.selectbox(
        "3차원 X축 속성",
        selected_features,
        index=0,
        format_func=lambda x: feature_dict[x],
        key="ax_x",
    )
  with col_y:
    axis_y = st.selectbox(
        "3차원 Y축 속성",
        selected_features,
        index=1,
        format_func=lambda x: feature_dict[x],
        key="ax_y",
    )
  with col_z:
    axis_z = st.selectbox(
        "3차원 Z축 속성",
        selected_features,
        index=2 if len(selected_features) > 2 else 0,
        format_func=lambda x: feature_dict[x],
        key="ax_z",
    )

  # 나머지 속성 중앙값 고정
  remaining_feats = [
      f for f in selected_features if f not in [axis_x, axis_y, axis_z]
  ]
  fixed_medians = {f: df_test[f].median() for f in remaining_feats}

  fig3d = go.Figure()

  # 1. 테스트 데이터 3D 산점도
  df_test_vis3d = df_test.copy()
  df_test_vis3d["stroke_label"] = df_test_vis3d["stroke"].map(
      {0: "아님 (0)", 1: "뇌졸중 (1)"}
  )

  for label, col in [("아님 (0)", "dodgerblue"), ("뇌졸중 (1)", "crimson")]:
    sub3d = df_test_vis3d[df_test_vis3d["stroke_label"] == label]
    fig3d.add_trace(
        go.Scatter3d(
            x=sub3d[axis_x],
            y=sub3d[axis_y],
            z=sub3d[axis_z],
            mode="markers",
            name=f"실제: {label}",
            marker=dict(size=4, color=col, opacity=0.7),
        )
    )

  # 2. 3D 격자 생성하여 로지스틱 회귀 0.5 등치면(Isosurface) 표시
  x_vals = np.linspace(df_test[axis_x].min(), df_test[axis_x].max(), 15)
  y_vals = np.linspace(df_test[axis_y].min(), df_test[axis_y].max(), 15)
  z_vals = np.linspace(df_test[axis_z].min(), df_test[axis_z].max(), 15)
  XX, YY, ZZ = np.meshgrid(x_vals, y_vals, z_vals)

  grid3d_dict = {axis_x: XX.ravel(), axis_y: YY.ravel(), axis_z: ZZ.ravel()}
  for f in remaining_feats:
    grid3d_dict[f] = np.full(XX.ravel().shape, fixed_medians[f])
  X_grid_3d = pd.DataFrame(grid3d_dict)

  prob_3d_b = lr_before.predict_proba(X_grid_3d)[:, 1]
  prob_3d_a = lr_after.predict_proba(X_grid_3d)[:, 1]

  # 가중치 전 0.5 평면
  fig3d.add_trace(
      go.Isosurface(
          x=XX.flatten(),
          y=YY.flatten(),
          z=ZZ.flatten(),
          value=prob_3d_b,
          isomin=0.48,
          isomax=0.52,
          surface_count=1,
          colorscale=[[0, "blue"], [1, "blue"]],
          opacity=0.3,
          name="가중치 전 0.5 평면",
          showscale=False,
      )
  )

  # 가중치 후 0.5 평면
  fig3d.add_trace(
      go.Isosurface(
          x=XX.flatten(),
          y=YY.flatten(),
          z=ZZ.flatten(),
          value=prob_3d_a,
          isomin=0.48,
          isomax=0.52,
          surface_count=1,
          colorscale=[[0, "orange"], [1, "orange"]],
          opacity=0.3,
          name="가중치 후 0.5 평면",
          showscale=False,
      )
  )

  fig3d.update_layout(
      title=f"3차원 결정 경계 평면 비교 ({feature_dict[axis_x]}, {feature_dict[axis_y]}, {feature_dict[axis_z]})",
      scene=dict(
          xaxis_title=feature_dict[axis_x],
          yaxis_title=feature_dict[axis_y],
          zaxis_title=feature_dict[axis_z],
      ),
      legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
      margin=dict(l=0, r=0, t=40, b=0),
  )

  st.plotly_chart(fig3d, use_container_width=True)
