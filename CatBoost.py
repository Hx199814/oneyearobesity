import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go # 引入Plotly用于绘制高级仪表盘

# 1. 页面配置：设置宽屏模式，自定义标题
st.set_page_config(
    page_title="学生健康风险智能评估系统",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 专业级CSS样式 (模仿SaaS后台风格)
st.markdown("""
<style>
    /* 全局背景色调 - 极简灰白 */
    .stApp {
        background-color: #f4f6f9;
    }
    
    /* 标题排版 */
    .main-title {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }

    /* 侧边栏美化 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.02);
    }
    
    /* 卡片容器通用样式 */
    .info-card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
        border: 1px solid #f1f5f9;
    }

    /* 建议框样式 */
    .advice-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #334155;
        margin-bottom: 10px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 5px;
    }
    .advice-content {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* 按钮样式重写 - 扁平化设计 */
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: background-color 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
    }
    
    /* 隐藏默认元素 */
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
        st.error("系统错误：未检测到模型文件 'CatBoost.pkl'。")
        return None

model = load_model()

# 4. 选项定义 (保持逻辑完全不变)
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

# 5. 计算逻辑 (保持逻辑完全不变)
def calculate_baseline_obesity(age, gender, height_cm, weight_kg):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    gender_code = 1 if gender == 1 else 0
    
    # ... (此处保留你原有的完整逻辑)
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

# --- 辅助函数：绘制高级仪表盘 ---
def create_gauge_chart(value, title, min_val, max_val, thresholds, suffix=""):
    """
    使用 Plotly 绘制高级仪表盘
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        number = {'suffix': suffix, 'font': {'size': 40, 'color': "#1e293b"}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 18, 'color': "#64748b"}},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2563eb", 'thickness': 0.25}, # 指针颜色
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': thresholds,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    # 更新布局，使其紧凑
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Arial"}
    )
    return fig

# 6. 主内容布局

# 顶部标题区
st.markdown('<div class="main-title">学生健康风险智能评估系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">基于 CatBoost 机器学习模型的多维度风险预测</div>', unsafe_allow_html=True)

# 7. 侧边栏：UI优化 + Form性能优化
with st.sidebar:
    st.markdown("### 📋 评估参数录入")
    st.markdown("请填写以下信息以生成评估报告：")
    
    with st.form(key='prediction_form'):
        
        # 模块1：基础生理指标 (使用列布局节省空间)
        st.markdown("**1. 基础生理指标**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            GENDER = st.selectbox("性别", options=list(GENDER_options.keys()), format_func=lambda x: GENDER_options[x])
            height_cm = st.number_input("身高 (cm)", 100.0, 220.0, 150.0, 1.0)
        with col_s2:
            AGE = st.selectbox("年龄", options=range(6, 19), format_func=lambda x: f"{x}岁")
            weight_kg = st.number_input("体重 (kg)", 20.0, 150.0, 45.0, 0.5)

        st.markdown("---")
        
        # 模块2：生活方式 (使用Expander保持整洁)
        with st.expander("2. 饮食与运动习惯", expanded=True):
            PEC = st.selectbox("每周体育课节数", list(PEC_options.keys()), format_func=lambda x: PEC_options[x])
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                FrFF = st.selectbox("吃水果频率", list(FrFF_options.keys()), format_func=lambda x: FrFF_options[x])
            with col_d2:
                DVT = st.selectbox("吃蔬菜种类", list(DVT_options.keys()), format_func=lambda x: DVT_options[x])
            HU = st.selectbox("长时间使用耳机", list(HU_options.keys()), format_func=lambda x: HU_options[x])

        # 模块3：心理与行为 (折叠，避免视觉干扰)
        with st.expander("3. 心理与行为问卷 (点击展开)", expanded=False):
            st.caption("请根据最近状况如实回答：")
            D1 = st.selectbox("小事也烦恼", list(D1_options.keys()), format_func=lambda x: D1_options[x])
            D2 = st.selectbox("食欲不振/不想吃", list(D2_options.keys()), format_func=lambda x: D2_options[x])
            D3 = st.selectbox("感到无法摆脱苦闷", list(D3_options.keys()), format_func=lambda x: D3_options[x])
            D9 = st.selectbox("觉得生活无意义", list(D9_options.keys()), format_func=lambda x: D9_options[x])
            D11 = st.selectbox("睡眠无法解乏", list(D11_options.keys()), format_func=lambda x: D11_options[x])
            D17 = st.selectbox("曾经放声痛哭", list(D17_options.keys()), format_func=lambda x: D17_options[x])
            st.markdown("---")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                FF = st.selectbox("打架记录", list(FF_options.keys()), format_func=lambda x: FF_options[x])
            with col_b2:
                PPP = st.selectbox("被责罚记录", list(PPP_options.keys()), format_func=lambda x: PPP_options[x])

        st.markdown("")
        submit_button = st.form_submit_button(label='开始智能评估', type="primary")

# 8. 预测结果区域
if submit_button:
    if model is not None:
        # --- 计算逻辑 ---
        baseline_obesity = calculate_baseline_obesity(AGE, GENDER, height_cm, weight_kg)
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        
        feature_values = [GENDER, baseline_obesity, D2, AGE, D1, D9, HU, D11, PEC, FrFF, D17, DVT, FF, D3, PPP]
        features = np.array([feature_values])
        
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]
        risk_probability = predicted_proba[1] * 100 # 转换为百分比

        # --- 可视化与布局 ---
        
        # 上半部分：数据仪表盘 (两列布局)
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            # 配置 BMI 仪表盘颜色带
            bmi_steps = [
                {'range': [0, 18.5], 'color': "#cbd5e1"}, # 偏瘦
                {'range': [18.5, 24], 'color': "#86efac"}, # 正常 (绿)
                {'range': [24, 28], 'color': "#fcd34d"},  # 超重 (黄)
                {'range': [28, 40], 'color': "#fca5a5"}   # 肥胖 (红)
            ]
            fig_bmi = create_gauge_chart(
                value=round(bmi, 1),
                title="当前 BMI 指数",
                min_val=10, max_val=35,
                thresholds=bmi_steps
            )
            st.plotly_chart(fig_bmi, use_container_width=True)
            # 文字补充状态
            status = "正常" if baseline_obesity == 0 else "偏高"
            color = "green" if baseline_obesity == 0 else "red"
            st.markdown(f"<div style='text-align:center; color:{color}; font-weight:bold;'>当前生理状态评估: {status}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_viz2:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            # 配置 风险概率 仪表盘颜色带
            risk_steps = [
                {'range': [0, 50], 'color': "#86efac"},  # 低风险 (绿)
                {'range': [50, 75], 'color': "#fcd34d"}, # 中风险 (黄)
                {'range': [75, 100], 'color': "#fca5a5"} # 高风险 (红)
            ]
            fig_risk = create_gauge_chart(
                value=round(risk_probability, 1),
                title="未来一年肥胖风险预测",
                min_val=0, max_val=100,
                thresholds=risk_steps,
                suffix="%"
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            risk_text = "低风险" if predicted_class == 0 else "高风险"
            risk_color_text = "green" if predicted_class == 0 else "red"
            st.markdown(f"<div style='text-align:center; color:{risk_color_text}; font-weight:bold;'>模型综合判定: {risk_text}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 下半部分：建议卡片
        st.markdown("### 🩺 个性化健康指导")
        
        col_advice1, col_advice2 = st.columns([1, 1])
        
        with col_advice1:
            st.markdown('<div class="info-card" style="min-height: 250px;">', unsafe_allow_html=True)
            if predicted_class == 1:
                st.markdown('<div class="advice-title" style="color:#ef4444;">⚠️ 重点干预建议</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="advice-content">
                <ul>
                    <li><strong>运动增强：</strong>仅依靠每周{pec}的体育课不足够，建议每日增加45分钟有氧运动。</li>
                    <li><strong>饮食预警：</strong>当前{veg}的蔬菜摄入量偏低，建议每餐增加一份绿叶蔬菜。</li>
                    <li><strong>心理调节：</strong>监测结果显示有潜在压力风险，建议家长多进行积极沟通。</li>
                </ul>
                </div>
                """.format(pec=PEC_options[PEC], veg=DVT_options[DVT]), unsafe_allow_html=True)
            else:
                st.markdown('<div class="advice-title" style="color:#22c55e;">🌟 保持与优化</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="advice-content">
                <ul>
                    <li><strong>维持现状：</strong>目前的BMI指数和生活习惯较为健康，请继续保持。</li>
                    <li><strong>预防为主：</strong>建议保持每周{pec}的体育锻炼频率。</li>
                    <li><strong>定期监测：</strong>建议每6个月复查一次身高体重。</li>
                </ul>
                </div>
                """.format(pec=PEC_options[PEC]), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_advice2:
            st.markdown('<div class="info-card" style="min-height: 250px;">', unsafe_allow_html=True)
            st.markdown('<div class="advice-title">💡 科学依据</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="advice-content">
            本评估基于大规模学生健康数据训练的 <strong>CatBoost 机器学习模型</strong>。<br><br>
            模型不仅考虑了身高体重等生理指标，还深度关联了<strong>情绪状态</strong>（如焦虑、睡眠质量）和<strong>行为习惯</strong>（如饮食结构、电子产品使用时长），为您提供更全面的健康风险预警。
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.error("无法加载模型，请检查服务器配置。")
else:
    # 初始欢迎界面
    st.markdown("""
    <div style="background-color:white; padding:40px; border-radius:10px; text-align:center; border:1px solid #e2e8f0;">
        <h3 style="color:#334155;">👋 欢迎使用健康评估系统</h3>
        <p style="color:#64748b;">请在左侧侧边栏填写学生详细信息，点击“开始智能评估”按钮获取详细的健康风险报告。</p>
    </div>
    """, unsafe_allow_html=True)
