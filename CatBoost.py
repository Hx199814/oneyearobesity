import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 1. 页面配置
st.set_page_config(
    page_title="学生肥胖风险评估系统",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS 样式 (调整了卡片的最小高度和列表的内边距，确保内容被包裹)
st.markdown("""
<style>
    /* 全局背景 */
    .stApp {
        background-color: #f4f6f9;
    }
    
    /* 标题样式 */
    .main-header {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    /* 侧边栏优化 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* 卡片容器样式 - 关键修改：增加 min-height 保证框体存在感 */
    .dashboard-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #f1f5f9;
        margin-bottom: 20px;
        min-height: 280px; /* 强制最小高度，防止内容溢出或框体折叠 */
    }
    
    /* 建议文本样式 */
    .advice-header {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
    }
    
    .advice-text {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.8;
    }
    
    .advice-text ul {
        margin-left: 0;
        padding-left: 20px; /* 确保列表符号有足够空间 */
        list-style-type: disc;
    }
    
    .advice-text li {
        margin-bottom: 8px;
    }
    
    /* 按钮样式 (保持不变) */
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        height: 45px;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 3. 加载模型
@st.cache_resource
def load_model():
    try:
        # 请确保 'CatBoost.pkl' 文件在运行目录下
        return joblib.load('CatBoost.pkl') 
    except FileNotFoundError:
        st.error("⚠️ 系统错误：未找到模型文件 'CatBoost.pkl'。")
        return None

model = load_model()

# 4. 定义选项字典 (保持不变)
GENDER_options = {1: '男生', 2: '女生'}
D2_options = {1: '没有或偶尔', 2: '有时', 3: '时常或一半时间', 4: '多数时间或持续', 5: '不清楚'}
D1_options = {1: '没有或偶尔', 2: '有时', 3: '时常或一半时间', 4: '多数时间或持续', 5: '不清楚'}
D9_options = {1: '没有或偶尔', 2: '有时', 3: '时常或一半时间', 4: '多数时间或持续', 5: '不清楚'}
HU_options = {1: '不会', 2: '会'}
D11_options = {1: '没有或偶尔', 2: '有时', 3: '时常或一半时间', 4: '多数时间或持续', 5: '不清楚'}
PEC_options = {1: '0节', 2: '1节', 3: '2节', 4: '3节', 5: '4节', 6: '5节及以上'}
FrFF_options = {1: '从来不吃', 2: '少于每天1次', 3: '每天1次', 4: '每天2次及以上'}
D17_options = {1: '没有或偶尔', 2: '有时', 3: '时常或一半时间', 4: '多数时间或持续', 5: '不清楚'}
DVT_options = {1: '从来不吃或少于每天1种', 2: '每天1种', 3: '每天2种', 4: '每天3次及以上'}
FF_options = {1: '是', 0: '否'}
D3_options = {1: '没有或偶尔', 2: '有时', 3: '时常或一半时间', 4: '多数时间或持续', 5: '不清楚'}
PPP_options = {1: '是', 0: '否'}

# 5. 核心计算逻辑 (保持不变)
def calculate_baseline_obesity(age, gender, height_cm, weight_kg):
    # ... (此处保留原有完整逻辑，为节省篇幅省略，请务必保留你原代码中的完整逻辑) ...
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    gender_code = 1 if gender == 1 else 0
    
    # 示例部分，请替换为你的完整 BMI 逻辑
    if age >= 18:
        if bmi >= 28.0: return 1
    elif age >= 6 and bmi >= 25.0:
        return 1
    return 0

# --- 辅助函数：创建仪表盘 ---
def create_gauge(value, title, min_val, max_val, steps, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': suffix, 'font': {'size': 35, 'color': "#334155"}},
        title={'text': title, 'font': {'size': 16, 'color': "#64748b"}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': "#2563eb", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': steps,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=220,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Arial"}
    )
    return fig

# 6. 主界面布局
st.markdown('<div class="main-header">学生肥胖风险智能评估系统</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748b; margin-bottom: 30px;">基于 CatBoost 机器学习模型的多维度健康风险预测平台</p>', unsafe_allow_html=True)

# 7. 侧边栏 (保持不变)
with st.sidebar:
    st.header("📋 评估参数录入")
    with st.form(key='prediction_form'):
        st.markdown("**1. 基础生理指标**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            GENDER = st.selectbox("性别", options=list(GENDER_options.keys()), format_func=lambda x: GENDER_options[x])
            height_cm = st.number_input("身高 (cm)", 100.0, 220.0, 150.0, 1.0)
        with col_s2:
            AGE = st.selectbox("年龄", options=range(6, 19), format_func=lambda x: f"{x}岁")
            weight_kg = st.number_input("体重 (kg)", 20.0, 150.0, 45.0, 0.5)
        st.markdown("---")
        with st.expander("2. 饮食与运动习惯", expanded=True):
            PEC = st.selectbox("每周体育课", list(PEC_options.keys()), format_func=lambda x: PEC_options[x])
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                FrFF = st.selectbox("吃水果频率", list(FrFF_options.keys()), format_func=lambda x: FrFF_options[x])
            with col_d2:
                DVT = st.selectbox("吃蔬菜种类", list(DVT_options.keys()), format_func=lambda x: DVT_options[x])
            HU = st.selectbox("耳机使用(>30分钟)", list(HU_options.keys()), format_func=lambda x: HU_options[x])
        with st.expander("3. 心理与行为问卷", expanded=False):
            D1 = st.selectbox("小事烦恼", list(D1_options.keys()), format_func=lambda x: D1_options[x])
            D2 = st.selectbox("食欲不振", list(D2_options.keys()), format_func=lambda x: D2_options[x])
            D3 = st.selectbox("无法摆脱苦闷", list(D3_options.keys()), format_func=lambda x: D3_options[x])
            D9 = st.selectbox("觉得生活无用", list(D9_options.keys()), format_func=lambda x: D9_options[x])
            D11 = st.selectbox("睡眠无法解乏", list(D11_options.keys()), format_func=lambda x: D11_options[x])
            D17 = st.selectbox("曾经痛哭", list(D17_options.keys()), format_func=lambda x: D17_options[x])
            st.divider()
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                FF = st.selectbox("打架记录", list(FF_options.keys()), format_func=lambda x: FF_options[x])
            with col_b2:
                PPP = st.selectbox("被责罚记录", list(PPP_options.keys()), format_func=lambda x: PPP_options[x])
        
        submit_button = st.form_submit_button(label='开始智能分析', type="primary")

# 8. 预测结果区域
if submit_button:
    if model is not None:
        # 计算逻辑
        baseline_obesity = calculate_baseline_obesity(AGE, GENDER, height_cm, weight_kg)
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        
        feature_values = [GENDER, baseline_obesity, D2, AGE, D1, D9, HU, D11, PEC, FrFF, D17, DVT, FF, D3, PPP]
        features = np.array([feature_values])
        
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]
        risk_probability = predicted_proba[1] * 100

        # Row 1: 仪表盘区域
        col_viz1, col_viz2 = st.columns(2)
        
        # 仪表盘 1: BMI
        with col_viz1:
            bmi_steps = [{'range': [0, 18.5], 'color': "#eff6ff"}, {'range': [18.5, 24], 'color': "#dcfce7"},
                         {'range': [24, 28], 'color': "#fef9c3"}, {'range': [28, 40], 'color': "#fee2e2"}]
            fig_bmi = create_gauge(round(bmi, 1), "当前 BMI 指数", 10, 35, bmi_steps)
            
            st.markdown('<div class="dashboard-card" style="min-height: 250px;">', unsafe_allow_html=True)
            st.plotly_chart(fig_bmi, use_container_width=True)
            status_text = "超重/肥胖" if baseline_obesity == 1 else "正常范围"
            status_color = "#ef4444" if baseline_obesity == 1 else "#22c55e"
            st.markdown(f"<div style='text-align:center; font-weight:bold; color:{status_color}; margin-top:-20px;'>当前生理状态: {status_text}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


        # 仪表盘 2: 风险概率
        with col_viz2:
            risk_steps = [{'range': [0, 50], 'color': "#dcfce7"}, {'range': [50, 75], 'color': "#fef9c3"}, {'range': [75, 100], 'color': "#fee2e2"}]
            fig_risk = create_gauge(round(risk_probability, 1), "未来一年肥胖风险预测率", 0, 100, risk_steps, "%")
            
            st.markdown('<div class="dashboard-card" style="min-height: 250px;">', unsafe_allow_html=True)
            st.plotly_chart(fig_risk, use_container_width=True)
            risk_text = "高风险" if predicted_class == 1 else "低风险"
            risk_color = "#ef4444" if predicted_class == 1 else "#22c55e"
            st.markdown(f"<div style='text-align:center; font-weight:bold; color:{risk_color}; margin-top:-20px;'>模型判定结果: {risk_text}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Row 2: 建议与说明 (重点修复：将 st.subheader 放在卡片渲染之前，并调整卡片内容)
        st.subheader("🩺 智能健康报告")
        
        col_advice1, col_advice2 = st.columns([1.3, 1])
        
        # 建议卡片 (左侧)
        with col_advice1:
            if predicted_class == 1:
                # 高风险建议 HTML 模板
                advice_html = f"""
                <div class="dashboard-card">
                    <div class="advice-header" style="color: #ef4444;">
                        ⚠️ 重点干预建议
                    </div>
                    <div class="advice-text">
                        <ul>
                            <li><strong>加强运动：</strong>您目前的体育课频率为 <strong>{PEC_options[PEC]}</strong>。建议在此基础上，每天增加至少45分钟的中等强度有氧运动（如慢跑、跳绳）。</li>
                            <li><strong>饮食调整：</strong>数据显示您的蔬菜摄入量为 <strong>{DVT_options[DVT]}</strong>，建议每餐增加一份绿叶蔬菜，并严格控制高热量零食。</li>
                            <li><strong>心理关注：</strong>模型检测到潜在的压力风险，建议家长关注孩子的情绪波动，保证充足睡眠。</li>
                        </ul>
                    </div>
                </div>
                """
                st.markdown(advice_html, unsafe_allow_html=True)
            else:
                # 低风险建议 HTML 模板
                advice_html = f"""
                <div class="dashboard-card">
                    <div class="advice-header" style="color: #22c55e;">
                        ✅ 保持与优化建议
                    </div>
                    <div class="advice-text">
                        <ul>
                            <li><strong>维持现状：</strong>恭喜！模型预测该学生未来一年的肥胖风险较低，请继续保持目前良好的生活习惯。</li>
                            <li><strong>持续运动：</strong>保持每周 <strong>{PEC_options[PEC]}</strong> 的体育锻炼频率，避免久坐。</li>
                            <li><strong>定期监测：</strong>建议每6个月测量一次身高体重，关注生长发育曲线。</li>
                        </ul>
                    </div>
                </div>
                """
                st.markdown(advice_html, unsafe_allow_html=True)

        # 说明卡片 (右侧)
        with col_advice2:
            explanation_html = """
            <div class="dashboard-card">
                <div class="advice-header" style="color: #334155;">
                    💡 评估说明
                </div>
                <div class="advice-text">
                    本系统基于 <strong>CatBoost 机器学习算法</strong>，综合分析了三个维度的风险因子：
                    <br><br>
                    1. <strong>生理指标</strong>：BMI 及基线肥胖状态。<br>
                    2. <strong>生活方式</strong>：饮食结构与运动频率。<br>
                    3. <strong>心理行为</strong>：焦虑水平、睡眠质量及冲动行为。
                    <br><br>
                    <small style="color:#94a3b8; font-size: 0.8rem;">* 预测结果仅供参考，不能替代专业医生的临床诊断。</small>
                </div>
            </div>
            """
            st.markdown(explanation_html, unsafe_allow_html=True)

    else:
        st.warning("⚠️ 模型文件未加载。")
else:
    st.markdown("""
    <div style="text-align:center; padding: 50px; color: #64748b; background: white; border-radius: 12px; border: 1px solid #e2e8f0; margin-top: 20px;">
        <h3>👋 欢迎使用</h3>
        <p>请在左侧侧边栏填写学生详细信息，并点击 <b>“开始智能分析”</b> 按钮。</p>
    </div>
    """, unsafe_allow_html=True)
