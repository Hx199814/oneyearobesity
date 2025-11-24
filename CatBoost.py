import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# 1. 页面配置：明确肥胖风险预测主题
st.set_page_config(
    page_title="学生肥胖风险预测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 自定义 CSS：保持原有质感，优化文字显示
st.markdown("""
    <style>
    /* 全局字体优化 */
    html, body, [class*="css"] {
        font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }
    
    /* 侧边栏背景微调 */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* 结果卡片样式 */
    .result-card {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #dc3545;
    }
    
    /* 调整 metric 样式 */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* 按钮样式优化 */
    div.stButton > button:first-child {
        background-color: #dc3545;
        color: white;
        border-radius: 8px;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #c82333;
        box-shadow: 0 4px 12px rgba(220,53,69,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# 加载模型
@st.cache_resource
def load_model():
    try:
        return joblib.load('CatBoost.pkl')
    except FileNotFoundError:
        st.warning("模型文件未找到，演示模式下将使用模拟数据")
        return None

model = load_model()

# --- 选项定义 ---
GENDER_options = {1: '男生', 2: '女生'}
D2_options = {1: '没有或偶尔', 2: '有时', 3: '时常或一半时间', 4: '多数时间或持续', 5: '不清楚'}
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
    st.markdown("## 📋 预测参数录入")
    st.markdown("请完善学生相关信息以评估肥胖风险")
    st.divider()

    # 基础生理指标（必选）
    with st.expander("基础生理指标", expanded=True):
        col1, col2 = st.columns(2)
        GENDER = col1.selectbox("性别", options=list(GENDER_options.keys()), format_func=lambda x: GENDER_options[x])
        AGE = col2.selectbox("年龄", options=list(range(6, 19)), format_func=lambda x: f"{x}岁")
        
        col3, col4 = st.columns(2)
        height_cm = col3.number_input("身高 (cm)", 100.0, 200.0, 150.0, 0.1)
        weight_kg = col4.number_input("体重 (kg)", 20.0, 100.0, 45.0, 0.1)
        
        # 实时 BMI 显示
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        baseline_obesity = calculate_baseline_obesity(AGE, GENDER, height_cm, weight_kg)
        
        st.markdown(
            f"""
            <div style='background-color:#e9ecef; padding:10px; border-radius:5px; font-size:0.9em; text-align:center; color:#495057'>
                当前 BMI 指数: <b>{bmi:.1f}</b>
            </div>
            """, unsafe_allow_html=True
        )

    # 生活方式与饮食
    with st.expander("生活方式与饮食", expanded=False):
        PEC = st.selectbox("每周体育课节数", options=list(PEC_options.keys()), format_func=lambda x: PEC_options[x])
        FrFF = st.selectbox("水果摄入频率", options=list(FrFF_options.keys()), format_func=lambda x: FrFF_options[x])
        DVT = st.selectbox("蔬菜摄入种类 (每天)", options=list(DVT_options.keys()), format_func=lambda x: DVT_options[x])

    # 心理健康状况
    with st.expander("心理健康状况", expanded=False):
        D1 = st.selectbox("受过往琐事困扰", options=list(D1_options.keys()), format_func=lambda x: D1_options[x])
        D2 = st.selectbox("食欲不振/胃口差", options=list(D2_options.keys()), format_func=lambda x: D2_options[x])
        D3 = st.selectbox("感到情绪低落/苦闷", options=list(D3_options.keys()), format_func=lambda x: D3_options[x])

    # 行为与其他
    with st.expander("行为与其他", expanded=False):
        HU = st.selectbox("长时间使用耳机 (>30分)", options=list(HU_options.keys()), format_func=lambda x: HU_options[x])
        FF = st.selectbox("过去12个月有打架行为", options=list(FF_options.keys()), format_func=lambda x: FF_options[x])
        PPP = st.selectbox("近期受到严厉责罚", options=list(PPP_options.keys()), format_func=lambda x: PPP_options[x])

# --- 主页面区域 ---
st.markdown("### 学生肥胖风险预测系统")
st.markdown("基于多维指标的智能肥胖风险评估工具")
st.divider()

# 预测逻辑
if st.button("开始肥胖风险预测", type="primary", use_container_width=True):
    if model is None:
        # 模拟预测（无模型时使用）
        st.warning("模型文件未加载，以下为模拟预测结果")
        # 基于BMI简单模拟风险
        if bmi >= 24:
            predicted_class = 1
            risk_score = np.random.uniform(60, 90)
        else:
            predicted_class = 0
            risk_score = np.random.uniform(10, 40)
        probability = risk_score if predicted_class == 1 else 100 - risk_score
    else:
        with st.spinner("正在计算肥胖风险..."):
            # 构建特征向量
            feature_values = [GENDER, baseline_obesity, D2, AGE, D1, D9_options[1], HU, D11_options[1], 
                              PEC, FrFF, D17_options[1], DVT, FF, D3, PPP]
            features = np.array([feature_values], dtype=np.float32)
            
            # 预测
            predicted_class = int(model.predict(features)[0])
            predicted_proba = model.predict_proba(features)[0]
            risk_score = predicted_proba[1] * 100  # 肥胖风险概率
            probability = risk_score if predicted_class == 1 else (100 - risk_score)
    
    # --- 结果展示区 ---
    col_metrics, col_viz = st.columns([1.2, 1])
    
    with col_metrics:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        if predicted_class == 1:
            st.markdown("#### 🔴 预测结果：高肥胖风险")
            st.markdown(f"该学生一年后的肥胖风险概率为 {risk_score:.1f}%"，需及时干预")
        else:
            st.markdown("#### 🟢 预测结果：低肥胖风险")
            st.markdown(f"该学生一年后的肥胖风险概率为 {risk_score:.1f}%"，风险可控")
        st.divider()
        st.metric("肥胖风险概率", f"{risk_score:.1f}%", 
                  delta="高于临界值" if risk_score > 50 else "低于临界值",
                  delta_color="inverse" if risk_score > 50 else "normal")
        st.markdown('</div>', unsafe_allow_html=True)

        # 简化建议部分
        st.subheader("💡 核心干预建议")
        if predicted_class == 1:
            st.markdown("""
            * 增加每日运动量，保证至少60分钟中高强度活动
            * 减少高热量食物摄入，增加蔬菜和水果比例
            * 控制静态活动时间
            """)
        else:
            st.markdown("""
            * 保持现有运动频率和健康饮食习惯
            * 定期监测身高体重变化
            """)

    with col_viz:
        # 肥胖风险仪表盘
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            title = {'text': "肥胖风险概率", 'font': {'size': 18, 'color': "#555"}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'suffix': "%", 'font': {'size': 26}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#999"},
                'bar': {'color': "#dc3545" if risk_score > 50 else "#28a745"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#f0f0f0",
                'steps': [
                    {'range': [0, 50], 'color': '#e8f5e9'},
                    {'range': [50, 100], 'color': '#ffebee'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}
            }
        ))
        fig.update_layout(
            height=350, 
            margin=dict(l=20, r=20, t=50, b=20),
            font={'family': "Arial"}
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    # 空状态提示
    st.markdown("""
    <div style="text-align: center; margin-top: 80px; color: #6c757d;">
        <h4>👈 请在左侧面板输入完整信息</h4>
        <p>点击上方按钮启动肥胖风险预测</p>
    </div>
    """, unsafe_allow_html=True)

# 页脚设计
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #adb5bd; font-size: 0.8rem;'>
    学生肥胖风险预测系统 © 2025 | Powered by CatBoost & Streamlit
</div>
""", unsafe_allow_html=True)
