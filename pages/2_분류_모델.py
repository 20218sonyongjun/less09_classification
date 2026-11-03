import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, _tree
import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실", page_icon="🤖", layout="wide"
)

# 데이터 로드
DATA_URL = "https://raw.githubusercontent.com/20218sonyongjun/data/refs/heads/main/stroke.csv"


@st.cache_data
def load_data():
  return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()

st.title("🤖 뇌졸중 예측 분류 모델 실습")
st.write(
    "데이터를 기반으로 로지스틱 회귀와 의사결정트리 모델을 학습하고, 예측"
    " 성능과 결정 경계를 시각화합니다."
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
default_features = [
    f for f in all_features if f != "bmi"
]  # 처음에는 체질량지수를 뺀 넷

selected_kr_names = st.multiselect(
    "모델 학습에 사용할 속성을 선택하세요 (2개 이상)",
    options=list(feature_dict.values()),
    default=[feature_dict[f] for f in default_features],
)

# 한글 이름을 원본 열 이름으로 매핑
kr_to_feature = {v: k for k, v in feature_dict.items()}
selected_features = [kr_to_feature[name] for name in selected_kr_names]

# 유효성 검사: 두 개보다 적게 고르면 안내문 표시 후 멈춤
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
# 번호 순으로 정렬
if "id" in df.columns:
  df_sorted = df.sort_values("id").reset_index(drop=True)
else:
  df_sorted = df.reset_index(drop=True)

# 10명씩 묶어 앞 3명을 테스트용, 나머지 7명을 학습용으로 고정 (테스트 총 1,533명)
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
# 3. 모델 학습 및 평가
# -----------------------------------------------------------------------------
# 1) 로지스틱 회귀 모델
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train))
lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test))

# 2) 의사결정트리 모델 (질문 3번까지만, 마지막 마디 5명 미만 금지, 난수 고정)
dt_model = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42
)
dt_model.fit(X_train, y_train)
dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt_model.predict(X_test))

# 3) 다수결 모델 (입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)
dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_model.fit(X_train, y_train)
dummy_train_acc = accuracy_score(y_train, dummy_model.predict(X_train))
dummy_test_acc = accuracy_score(y_test, dummy_model.predict(X_test))

st.subheader("📊 모델별 정확도 비교")
st.write(
    f"총 테스트 데이터 수: **{len(df_test):,}명** (요건에 따른 1,533명 고정)"
)

col1, col2, col3 = st.columns(3)

with col1:
  st.markdown("#### 로지스틱 회귀<br>(확률로 답하는 모델)", unsafe_allow_html=True)
  st.metric(label="테스트 정확도", value=f"{lr_test_acc*100:.2f}%")
  st.markdown(
      f"훈련 데이터: {lr_train_acc*100:.2f}% | 테스트 데이터:"
      f" **{lr_test_acc*100:.2f}%**"
  )

with col2:
  st.markdown("#### 의사결정트리<br>(질문으로 답하는 모델)", unsafe_allow_html=True)
  st.metric(label="테스트 정확도", value=f"{dt_test_acc*100:.2f}%")
  st.markdown(
      f"훈련 데이터: {dt_train_acc*100:.2f}% | 테스트 데이터:"
      f" **{dt_test_acc*100:.2f}%**"
  )

with col3:
  st.markdown(
      "#### 다수결 모델<br>(입력 안 보는 모델)", unsafe_allow_html=True
  )
  st.metric(label="테스트 정확도", value=f"{dummy_test_acc*100:.2f}%")
  st.markdown(
      f"훈련 데이터: {dummy_train_acc*100:.2f}% | 테스트 데이터:"
      f" **{dummy_test_acc*100:.2f}%**"
  )

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. 결정 경계 및 산점도 시각화 (Plotly)
# -----------------------------------------------------------------------------
st.subheader("📈 결정 경계 및 테스트 데이터 산점도")

# 고른 속성 가운데 둘을 골라 가로축과 세로축으로 삼기
vis_x_col = st.selectbox(
    "가로축 속성",
    selected_features,
    index=0,
    format_func=lambda x: feature_dict[x],
    key="vis_x",
)
vis_y_col = st.selectbox(
    "세로축 속성",
    selected_features,
    index=1 if len(selected_features) > 1 else 0,
    format_func=lambda x: feature_dict[x],
    key="vis_y",
)

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산
other_features = [
    f for f in selected_features if f not in [vis_x_col, vis_y_col]
]
other_medians = {}
for f in other_features:
  other_medians[f] = df_test[f].median()

if other_features:
  median_texts = [
      f"**{feature_dict[f]}**: {other_medians[f]:.2f}" for f in other_features
  ]
  st.info(
      "📌 두 축이 아닌 속성은 **테스트 데이터의 중앙값**에 세워 두고"
      f" 계산했습니다: {', '.join(median_texts)}"
  )
else:
  st.info(
      "📌 선택한 속성이 정확히 2개이므로 추가로 고정된 속성이 없습니다."
  )

# 격자(Grid) 생성
x_min, x_max = df_test[vis_x_col].min(), df_test[vis_x_col].max()
y_min, y_max = df_test[vis_y_col].min(), df_test[vis_y_col].max()
x_margin = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0

xx, yy = np.meshgrid(
    np.linspace(x_min - x_margin, x_max + x_margin, 100),
    np.linspace(y_min - y_margin, y_max + y_margin, 100),
)

grid_dict = {}
for f in selected_features:
  if f == vis_x_col:
    grid_dict[f] = xx.ravel()
  elif f == vis_y_col:
    grid_dict[f] = yy.ravel()
  else:
    grid_dict[f] = np.full(xx.ravel().shape, other_medians[f])

X_grid = pd.DataFrame(grid_dict)

# 의사결정트리 영역 예측 (배경 색상)
dt_grid_pred = dt_model.predict(X_grid).reshape(xx.shape)

# 로지스틱 회귀 확률 예측 (0.5 경계선)
if hasattr(lr_model, "predict_proba"):
  lr_grid_prob = lr_model.predict_proba(X_grid)[:, 1].reshape(xx.shape)
else:
  lr_grid_prob = lr_model.decision_function(X_grid).reshape(xx.shape)

p_min, p_max = lr_grid_prob.min(), lr_grid_prob.max()
boundary_in_range = (p_min <= 0.5) and (p_max >= 0.5)

if not boundary_in_range:
  st.warning(
      "⚠️ 로지스틱 회귀 모델의 0.5로 가르는 선(결정 경계선)이 현재 그림 범위"
      " 밖입니다."
  )
else:
  st.success(
      "✅ 로지스틱 회귀 모델의 0.5로 가르는 선이 그림 범위 내에 있습니다."
  )

# Plotly 시각화
fig = go.Figure()

# 1. 의사결정트리 영역 옅은 색 칠하기
fig.add_trace(
    go.Contour(
        z=dt_grid_pred,
        x=np.linspace(x_min - x_margin, x_max + x_margin, 100),
        y=np.linspace(y_min - y_margin, y_max + y_margin, 100),
        colorscale=[[0, "rgba(173, 216, 230, 0.15)"], [1, "rgba(255, 182, 193, 0.15)"]],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 영역",
    )
)

# 2. 로지스틱 회귀 0.5 결정 경계선
if boundary_in_range:
  fig.add_trace(
      go.Contour(
          z=lr_grid_prob,
          x=np.linspace(x_min - x_margin, x_max + x_margin, 100),
          y=np.linspace(y_min - y_margin, y_max + y_margin, 100),
          contours=dict(
              start=0.5, end=0.5, size=0.1, coloring="lines", showlabels=False
          ),
          line=dict(color="darkblue", width=3, dash="dash"),
          showscale=False,
          name="로지스틱 경계선 (p=0.5)",
      )
  )

# 3. 테스트 데이터 산점도 (실제 뇌졸중 여부 색상 구분)
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
          name=f"테스트 데이터: {label}",
          marker=dict(color=col, size=8, opacity=0.85, line=dict(width=0.5, color="white")),
      )
  )

fig.update_layout(
    title=f"분류 모델 결정 경계 및 테스트 데이터 산점도 ({feature_dict[vis_x_col]} vs {feature_dict[vis_y_col]})",
    xaxis_title=feature_dict[vis_x_col],
    yaxis_title=feature_dict[vis_y_col],
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(l=40, r=40, t=50, b=40),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. 의사결정트리 질문 가지 구조도 (Graphviz DOT)
# -----------------------------------------------------------------------------
st.subheader("🌳 의사결정트리 질문 가지 구조도")


def generate_tree_dot(tree, feat_names, feat_dict):
  tree_ = tree.tree_
  dot_lines = []
  dot_lines.append("digraph Tree {")
  dot_lines.append('node [shape=box, style="filled, rounded", fontname="sans-serif", fontsize=11];')
  dot_lines.append('edge [fontname="sans-serif", fontsize=10];')

  leaf_cnt = 0
  not_stroke_leaf_cnt = 0
  stroke_leaf_cnt = 0
  used_feats = set()

  def recurse(node, parent_id, branch_lbl):
    nonlocal leaf_cnt, not_stroke_leaf_cnt, stroke_leaf_cnt, used_feats

    samples = int(tree_.n_node_samples[node])
    values = tree_.value[node][0]
    non_stroke_c = int(values[0])
    stroke_c = int(values[1]) if len(values) > 1 else 0
    ratio = (stroke_c / samples * 100) if samples > 0 else 0.0

    if tree_.feature[node] != _tree.TREE_UNDEFINED:
      feat_orig = feat_names[tree_.feature[node]]
      feat_kr = feat_dict.get(feat_orig, feat_orig)
      used_feats.add(feat_kr)
      thresh = tree_.threshold[node]
      thresh_str = f"{int(thresh)}" if thresh.is_integer() else f"{thresh:.2f}"

      lbl = f"{feat_kr} <= {thresh_str}\\n사람 수: {samples}명\\n뇌졸중: {stroke_c}명 ({ratio:.1f}%)"
      color = "#e5f5f9"
    else:
      leaf_cnt += 1
      is_stroke = stroke_c > non_stroke_c
      if is_stroke:
        stroke_leaf_cnt += 1
        color = "#fee0d2"  # 연한 빨강 (뇌졸중)
        res_str = "답: 뇌졸중(1)"
      else:
        not_stroke_leaf_cnt += 1
        color = "#e5f5f9"  # 연한 파랑 (아님)
        res_str = "답: 아님(0)"

      lbl = f"{res_str}\\n사람 수: {samples}명\\n뇌졸중: {stroke_c}명 ({ratio:.1f}%)"

    node_id = f"node_{node}"
    dot_lines.append(f'  {node_id} [label="{lbl}", fillcolor="{color}"];')

    if parent_id is not None:
      dot_lines.append(f'  {parent_id} -> {node_id} [label="{branch_lbl}"];')

    left = tree_.children_left[node]
    right = tree_.children_right[node]

    if left != _tree.TREE_LEAF:
      recurse(left, node_id, "예")
      recurse(right, node_id, "아니요")

  recurse(0, None, "")
  dot_lines.append("}")
  return (
      "\n".join(dot_lines),
      leaf_cnt,
      not_stroke_leaf_cnt,
      stroke_leaf_cnt,
      list(used_feats),
  )


dot_str, leaf_total, not_stroke_l, stroke_l, used_feature_list = generate_tree_dot(
    dt_model, selected_features, feature_dict
)

st.graphviz_chart(dot_str)

st.markdown("### 📋 의사결정트리 분석 요약")
st.write(f"- **답을 내는 마디가 모두 몇 칸인가요?**: 총 **{leaf_total}개**")
st.write(
    f"- **그중 몇 칸이 아님이라고 답하나요?**: 총 **{not_stroke_l}개** (뇌졸중이라고"
    f" 답하는 마디: {stroke_l}개)"
)
if used_feature_list:
  st.write(
      "- **고른 속성 가운데 이 나무가 실제로 물은 것은 무엇인가요?**: "
      + ", ".join([f"**{f}**" for f in used_feature_list])
  )
else:
  st.write("- **고른 속성 가운데 이 나무가 실제로 물은 것은 무엇인가요?**: 없음")
