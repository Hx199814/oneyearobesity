import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# 1. 页面配置：明确为肥胖风险预测
st.set_page_config(
    page_title="学生肥胖风险预测系统",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 自定义 CSS (保持美观，微调样式)
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    .result-card {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #007bff;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem;
        font-weight: 700;
    }
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 12px rgba(0,123,255,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# 加载模型
@st.cache_resource
def load_model():
    try:
        return joblib.load('CatBoost.pkl')
    except FileNotFoundError:
        return None

model = load_model()

# --- 选项定义 ---
GENDER_options = {1: '男生', 2: '女生'}
D2_options = {1: '没有或偶尔', 2: '有时', 3: '时常或一半时间', 4: '多数时间或持续', 5: '不清楚'}
# 复用选项
D1_options = D2_options
D9_options = D2_options
HU_options = {1: '不会', 2: '会'}
D11_options = D2_options
PEC_options = {1: '0节', 2: '1节', 3: '2节', 4: '3节', 5: '4节', 6: '5节及以上'}
FrFF_options = {1: '从来不吃', 2: '少于每天1次', 3: '每天1次', 4: '每天2次及以上'}
D17_options = D2_options
DVT_options = {1: '从来不吃或少于每天1种', 2: '每天1种', 3: '每天2种', 4: '每天3次及以上'}
FF_options = {1: '是', 0: '否'}
D3_options = D2_options
PPP_options = {1: '是', 0: '否'}

# 计算基线肥胖函数
def calculate_baseline_obesity(age, gender, height_cm, weight_kg):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    gender_code = 1 if gender == 1 else 0
    
    thresholds = {
        6: (17.7, 17.5), 6.5: (18.1, 18.0), 7: (18.7, 18.5), 7.5: (19.2, 19.0),
        8: (19.7, 19.4), 8.5: (20.3, 19.9), 9: (20.8, 20.4), 9.5: (21.4, 21.0),
        10: (21.9, 21.5), 10.5: (22.5, 22.1), 11: (23.0, 22.7), 11.5: (23.6, 23.3),
        12: (24.1, 23.9), 12.5: (24.7, 24.5), 13: (25.2, 25.6), 13.5: (25.7, 25.6),
        14: (26.1, 25.9), 14.5: (26.4, 26.3), 15: (26.6, 26.6), 15.5: (26.9, 26.9),
        16: (27.1, 27.1), 16.5: (27.4, 27.4), 17: (27.6, 27.6), 17.5: (27.8, 27.8),
        18: (28.0, 28.0)
    }
    
    lookup_age = int(age * 2) / 2
    if lookup_age >= 18: lookup_age = 18
    if lookup_age < 6: lookup_age = 6
    
    if lookup_age in thresholds:
        limit = thresholds[lookup_age][0] if gender_code == 1 else thresholds[lookup_age][1]
        return 1 if bmi >= limit else 0
    return 0 if bmi < 28.0 else 1

# --- 侧边栏设计 ---
with st.sidebar:
    st.markdown("## 📋 学生信息录入")
    st.divider()

    with st.expander("基础生理指标", expanded=True):
        col1, col2 = st.columns(2)
        GENDER = col1.selectbox("性别", options=list(GENDER_options.keys()), format_func=lambda x: GENDER_options[x])
        AGE = col2.selectbox("年龄", options=list(range(6, 19)), format_func=lambda x: f"{x}岁")
        
        col3, col4 = st.columns(2)
        height_cm = col3.number_input("身高 (cm)", 100.0, 200.0, 150.0, 0.1)
        weight_kg = col4.number_input("体重 (kg)", 20.0, 100.0, 45.0, 0.1)
        
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        baseline_obesity = calculate_baseline_obesity(AGE, GENDER, height_cm, weight_kg)
        
        st.markdown(f"<div style='text-align:center; color:#666; font-size:0.9em;'>当前 BMI: <b>{bmi:.1f}</b></div>", unsafe_allow_html=True)

    with st.expander("生活与饮食"):
        PEC = st.selectbox("每周体育课节数", options=list(PEC_options.keys()), format_func=lambda x: PEC_options[x])
        FrFF = st.selectbox("过去七天吃新鲜水果次数", options=list(FrFF_options.keys()), format_func=lambda x: FrFF_options[x])
        DVT = st.selectbox("每天吃几种蔬菜", options=list(DVT_options.keys()), format_func=lambda x: DVT_options[x])

    with st.expander("情绪状态（最近一周）"):
        D1 = st.selectbox("以前从不困扰我的事情现在让我烦恼", options=list(D1_options.keys()), format_func=lambda x: D1_options[x])
        D2 = st.selectbox("我不想吃东西；我胃口不好", options=list(D2_options.keys()), format_func=lambda x: D2_options[x])
        D3 = st.selectbox("我觉得即便有家人或朋友帮助也无法摆脱这种苦闷", options=list(D3_options.keys()), format_func=lambda x: D3_options[x])
        D9 = st.selectbox("我认为我的生活一无是处", options=list(D9_options.keys()), format_func=lambda x: D9_options[x])
        D11 = st.selectbox("我睡觉后不能缓解疲劳", options=list(D11_options.keys()), format_func=lambda x: D11_options[x])
        D17 = st.selectbox("我曾经放声痛哭", options=list(D17_options.keys()), format_func=lambda x: D17_options[x])

    with st.expander("其他行为"):
        HU = st.selectbox("是否使用耳机（至少连续30分钟）", options=list(HU_options.keys()), format_func=lambda x: HU_options[x])
        FF = st.selectbox("过去12个月里是否与他人动手打架", options=list(FF_options.keys()), format_func=lambda x: FF_options[x])
        PPP = st.selectbox("过去30天是否曾被家长打骂", options=list(PPP_options.keys()), format_func=lambda x: PPP_options[x])

# --- 主页面区域 ---

st.markdown("### ⚖️ 学生肥胖风险预测系统")
st.markdown("基于机器学习模型预测学生未来肥胖风险概率")
st.divider()

if st.button("开始预测分析", type="primary", use_container_width=True):
    if model is None:
        st.error("错误：模型文件未找到。")
    else:
        with st.spinner("正在计算风险..."):
            try:
                feature_values = [GENDER, baseline_obesity, D2, AGE, D1, D9, HU, D11, PEC, FrFF, D17, DVT, FF, D3, PPP]
                features = np.array([feature_values], dtype=np.float32)
                
                predicted_proba = model.predict_proba(features)[0]
                # 核心修改：直接获取“是肥胖(Class 1)”的概率
                obesity_risk_score = predicted_proba[1] * 100
                
                col_metrics, col_viz = st.columns([1.2, 1])
                
                with col_metrics:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    # 简化逻辑：超过50%即为高风险
                    if obesity_risk_score >= 50:
                        st.markdown("#### 🔴 预测结果：高风险")
                        st.metric("肥胖风险概率", f"{obesity_risk_score:.1f}%", delta="注意", delta_color="inverse")
                        st.markdown("---")
                        st.markdown("**💡 改善建议：**")
                        st.markdown("""
                        1. **增加运动**：每日中高强度运动至少60分钟。
                        2. **控制饮食**：减少糖分摄入，增加蔬菜比例。
                        3. **规律作息**：保证充足睡眠，避免熬夜。
                        """)
                    else:
                        st.markdown("#### 🟢 预测结果：低风险")
                        # 虽然是低风险，也显示肥胖风险概率（数值会很低，例如 10%），delta 显示为绿色表示“安全”
                        st.metric("肥胖风险概率", f"{obesity_risk_score:.1f}%", delta="-低风险", delta_color="normal")
                        st.markdown("---")
                        st.markdown("**💡 保持建议：**")
                        st.markdown("""
                        1. **维持现状**：继续保持良好的饮食和运动习惯。
                        2. **定期监测**：每季度记录一次身高体重变化。
                        """)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_viz:
                    # 仪表盘直接显示“肥胖风险”
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = obesity_risk_score,
                        title = {'text': "肥胖风险值", 'font': {'size': 20, 'color': "#333"}},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        number = {'suffix': "%", 'font': {'size': 30}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': "#dc3545" if obesity_risk_score >= 50 else "#28a745"},
                            'bgcolor': "white",
                            'steps': [
                                {'range': [0, 50], 'color': '#f0fdf4'}, # 浅绿背景
                                {'range': [50, 100], 'color': '#fef2f2'} # 浅红背景
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50}
                        }
                    ))
                    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"预测出错: {str(e)}")
else:
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #666;">
        <h4>👈 请在左侧填写信息</h4>
        <p>点击上方按钮获取肥胖风险分析报告</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("学生肥胖风险预测系统 © 2025")
