import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # 引入Plotly用于绘制高级仪表盘

# 1. 页面配置：设置宽屏模式
st.set_page_config(
    page_title="学生健康风险智能评估系统",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 高级UI样式 (CSS) - 模拟SaaS软件风格
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
    
    /* 卡片容器样式 */
    .dashboard-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border: 1px solid #f1f5f9;
        margin-bottom: 20px;
    }
    
    /* 建议文本样式 */
    .advice-header {
        font-weight: 600;
        font-size: 1.1rem;
        color: #334155;
        margin-bottom: 10px;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 8px;
    }
    .advice-text {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* 按钮样式 */
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        height: 45px;
        font-weight: 600;
        transition: all 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
    }
    
    /* 隐藏多余元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 3. 加载模型
@st.cache_resource
def load_model():
    try:
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
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    gender_code = 1 if gender == 1 else 0
    
    if age >= 6 and age < 6.5:
        if gender_code == 1 and bmi >= 17.7: return 1
        elif gender_code == 0 and bmi >= 17.5: return 1
    elif age >= 6.5 and age < 7:
        if gender_code == 1 and bmi >= 18.1: return 1
        elif gender_code == 0 and bmi >= 18.0: return 1
    elif age >= 7 and age < 7.5:
        if gender_code == 1 and bmi >= 18.7: return 1
        elif gender_code == 0 and bmi >= 18.5: return 1
    elif age >= 7.5 and age < 8:
        if gender_code == 1 and bmi >= 19.2: return 1
        elif gender_code == 0 and bmi >= 19.0: return 1
    elif age >= 8 and age < 8.5:
        if gender_code == 1 and bmi >= 19.7: return 1
        elif gender_code == 0 and bmi >= 19.4: return 1
    elif age >= 8.5 and age < 9:
        if gender_code == 1 and bmi >= 20.3: return 1
        elif gender_code == 0 and bmi >= 19.9: return 1
    elif age >= 9 and age < 9.5:
        if gender_code == 1 and bmi >= 20.8: return 1
        elif gender_code == 0 and bmi >= 20.4: return 1
    elif age >= 9.5 and age < 10:
        if gender_code == 1 and bmi >= 21.4: return 1
        elif gender_code == 0 and bmi >= 21.0: return 1
    elif age >= 10 and age < 10.5:
        if gender_code == 1 and bmi >= 21.9: return 1
        elif gender_code == 0 and bmi >= 21.5: return 1
    elif age >= 10.5 and age < 11:
        if gender_code == 1 and bmi >= 22.5: return 1
        elif gender_code == 0 and bmi >= 22.1: return 1
    elif age >= 11 and age < 11.5:
        if gender_code == 1 and bmi >= 23.0: return 1
        elif gender_code == 0 and bmi >= 22.7: return 1
    elif age >= 11.5 and age < 12:
        if gender_code == 1 and bmi >= 23.6: return 1
        elif gender_code == 0 and bmi >= 23.3: return 1
    elif age >= 12 and age < 12.5:
        if gender_code == 1 and bmi >= 24.1: return 1
        elif gender_code == 0 and bmi >= 23.9: return 1
    elif age >= 12.5 and age < 13:
        if gender_code == 1 and bmi >= 24.7: return 1
        elif gender_code == 0 and bmi >= 24.5: return 1
    elif age >= 13 and age < 13.5:
        if gender_code == 1 and bmi >= 25.2: return 1
        elif gender_code == 0 and bmi >= 25.6: return 1
    elif age >= 13.5 and age < 14:
        if gender_code == 1 and bmi >= 25.7: return 1
        elif gender_code == 0 and bmi >= 25.6: return 1
    elif age >= 14 and age < 14.5:
        if gender_code == 1 and bmi >= 26.1: return 1
        elif gender_code == 0 and bmi >= 25.9: return 1
    elif age >= 14.5 and age < 15:
        if gender_code == 1 and bmi >= 26.4: return 1
        elif gender_code == 0 and bmi >= 26.3: return 1
    elif age >= 15 and age < 15.5:
        if gender_code == 1 and bmi >= 26.6: return 1
        elif gender_code == 0 and bmi >= 26.6: return 1
    elif age >= 15.5 and age < 16:
        if gender_code == 1 and bmi >= 26.9: return 1
        elif gender_code == 0 and bmi >= 26.9: return 1
    elif age >= 16 and age < 16.5:
        if gender_code == 1 and bmi >= 27.1: return 1
        elif gender_code == 0 and bmi >= 27.1: return 1
    elif age >= 16.5 and age < 17:
        if gender_code == 1 and bmi >= 27.4: return 1
        elif gender_code == 0 and bmi >= 27.4: return 1
    elif age >= 17 and age < 17.5:
        if gender_code == 1 and bmi >= 27.6: return 1
        elif gender_code == 0 and bmi >= 27.6: return 1
    elif age >= 17.5 and age < 18:
        if gender_code == 1 and bmi >= 27.8: return 1
        elif gender_code == 0 and bmi >= 27.8: return 1
    elif age >= 18:
        if bmi >= 28.0: return 1
    
    return 0

# --- 辅助函数：创建仪表盘 (Gauge Chart) ---
def create_gauge(value, title, min_val, max_val, steps, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': suffix, 'font': {'size': 35, 'color': "#334155"}},
        title={'text': title, 'font': {'size': 16, 'color': "#64748b"}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': "#2563eb", 'thickness': 0.25}, # 指针颜色
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
        paper_bgcolor='rgba(0,0,0,0)', # 透明背景
        font={'family': "Arial"}
    )
    return fig

# 6. 主界面布局
st.markdown('<div class="main-header">学生肥胖风险智能评估系统</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748b; margin-bottom: 30px;">基于 CatBoost 机器学习模型的多维度健康风险预测平台</p>', unsafe_allow_html=True)

# 7. 侧边栏：优化布局与分组
with st.sidebar:
    st.header("📋 评估参数录入")
    
    # 使用 st.form 解决卡顿问题：所有输入填完后，点按钮才刷新
    with st.form(key='prediction_form'):
        
        # 模块1：生理指标 (两列布局，更紧凑)
        st.markdown("**1. 基础生理指标**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            GENDER = st.selectbox("性别", options=list(GENDER_options.keys()), format_func=lambda x: GENDER_options[x])
            height_cm = st.number_input("身高 (cm)", 100.0, 220.0, 150.0, 1.0)
        with col_s2:
            AGE = st.selectbox("年龄", options=range(6, 19), format_func=lambda x: f"{x}岁")
            weight_kg = st.number_input("体重 (kg)", 20.0, 150.0, 45.0, 0.5)
        
        st.markdown("---")

        # 模块2：生活习惯 (使用 Expander 折叠，保持界面清爽)
        with st.expander("2. 饮食与运动习惯", expanded=True):
            PEC = st.selectbox("每周体育课", list(PEC_options.keys()), format_func=lambda x: PEC_options[x])
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                FrFF = st.selectbox("吃水果频率", list(FrFF_options.keys()), format_func=lambda x: FrFF_options[x])
            with col_d2:
                DVT = st.selectbox("吃蔬菜种类", list(DVT_options.keys()), format_func=lambda x: DVT_options[x])
            HU = st.selectbox("耳机使用(>30分钟)", list(HU_options.keys()), format_func=lambda x: HU_options[x])

        # 模块3：心理与行为 (默认折叠，因为问题较多)
        with st.expander("3. 心理与行为问卷 (点击展开)", expanded=False):
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

        # 提交按钮
        submit_button = st.form_submit_button(label='开始智能分析', type="primary")

# 8. 预测与结果显示区域
if submit_button:
    if model is not None:
        # --- 保持原有的计算逻辑 ---
        baseline_obesity = calculate_baseline_obesity(AGE, GENDER, height_cm, weight_kg)
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        
        feature_values = [GENDER, baseline_obesity, D2, AGE, D1, D9, HU, D11, PEC, FrFF, D17, DVT, FF, D3, PPP]
        features = np.array([feature_values])
        
        # 预测
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]
        risk_probability = predicted_proba[1] * 100 # 转为百分比

        # --- 可视化展示层 (Dashboard 风格) ---
        
        # 第一部分：仪表盘区域 (Row 1)
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            # BMI 仪表盘配置
            bmi_steps = [
                {'range': [0, 18.5], 'color': "#eff6ff"},  # 偏瘦 (淡蓝)
                {'range': [18.5, 24], 'color': "#dcfce7"}, # 正常 (淡绿)
                {'range': [24, 28], 'color': "#fef9c3"},  # 超重 (淡黄)
                {'range': [28, 40], 'color': "#fee2e2"}   # 肥胖 (淡红)
            ]
            fig_bmi = create_gauge(
                value=round(bmi, 1), 
                title="当前 BMI 指数", 
                min_val=10, max_val=35, 
                steps=bmi_steps
            )
            st.plotly_chart(fig_bmi, use_container_width=True)
            
            # 状态文字
            status_text = "超重/肥胖" if baseline_obesity == 1 else "正常范围"
            status_color = "#ef4444" if baseline_obesity == 1 else "#22c55e"
            st.markdown(f"<div style='text-align:center; font-weight:bold; color:{status_color}'>当前生理状态: {status_text}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_viz2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            # 风险概率仪表盘配置
            risk_steps = [
                {'range': [0, 50], 'color': "#dcfce7"},   # 低风险
                {'range': [50, 75], 'color': "#fef9c3"},  # 中风险
                {'range': [75, 100], 'color': "#fee2e2"}  # 高风险
            ]
            fig_risk = create_gauge(
                value=round(risk_probability, 1), 
                title="未来一年肥胖风险预测概率", 
                min_val=0, max_val=100, 
                steps=risk_steps, 
                suffix="%"
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # 风险结论
            risk_text = "高风险" if predicted_class == 1 else "低风险"
            risk_color = "#ef4444" if predicted_class == 1 else "#22c55e"
            st.markdown(f"<div style='text-align:center; font-weight:bold; color:{risk_color}'>模型判定结果: {risk_text}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 第二部分：分析与建议 (Row 2)
        st.subheader("🩺 智能健康报告")
        
        col_advice1, col_advice2 = st.columns([1.2, 1])
        
        with col_advice1:
            st.markdown('<div class="dashboard-card" style="min-height: 250px;">', unsafe_allow_html=True)
            if predicted_class == 1:
                st.markdown('<div class="advice-header" style="color: #ef4444;">⚠️ 重点干预建议</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="advice-text">
                <ul>
                    <li><strong>加强运动：</strong>您目前的体育课频率为 <strong>{PEC_options[PEC]}</strong>。建议在此基础上，每天增加至少45分钟的中等强度有氧运动（如慢跑、跳绳）。</li>
                    <li><strong>饮食调整：</strong>数据显示您的蔬菜摄入量为 <strong>{DVT_options[DVT]}</strong>，建议每餐增加一份绿叶蔬菜，并严格控制高热量零食。</li>
                    <li><strong>心理关注：</strong>模型检测到潜在的压力风险，建议家长关注孩子的情绪波动，保证充足睡眠。</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="advice-header" style="color: #22c55e;">✅ 保持与优化建议</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="advice-text">
                <ul>
                    <li><strong>维持现状：</strong>恭喜！模型预测该学生未来一年的肥胖风险较低，请继续保持目前良好的生活习惯。</li>
                    <li><strong>持续运动：</strong>保持每周 <strong>{PEC_options[PEC]}</strong> 的体育锻炼频率，避免久坐。</li>
                    <li><strong>定期监测：</strong>建议每6个月测量一次身高体重，关注生长发育曲线。</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_advice2:
            st.markdown('<div class="dashboard-card" style="min-height: 250px;">', unsafe_allow_html=True)
            st.markdown('<div class="advice-header">💡 评估说明</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="advice-text">
            本系统基于 <strong>CatBoost 机器学习算法</strong>，综合分析了三个维度的风险因子：
            1.  <strong>生理指标</strong>：BMI 及基线肥胖状态。
            2.  <strong>生活方式</strong>：饮食结构与运动频率。
            3.  <strong>心理行为</strong>：焦虑水平、睡眠质量及冲动行为。
            <br><br>
            <small style="color:#94a3b8">* 预测结果仅供参考，不能替代专业医生的临床诊断。</small>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("⚠️ 模型文件未加载，请检查服务器配置。")
else:
    # 初始欢迎状态
    st.markdown("""
    <div style="text-align:center; padding: 50px; color: #64748b; background: white; border-radius: 12px; border: 1px solid #e2e8f0;">
        <h3>👋 欢迎使用</h3>
        <p>请在左侧侧边栏填写学生详细信息，并点击 <b>“开始智能分析”</b> 按钮。</p>
    </div>
    """, unsafe_allow_html=True)
