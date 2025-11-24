import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt # 使用Altair替代Matplotlib，更高级、更快

# 1. 页面配置：设置宽屏模式
st.set_page_config(
    page_title="学生健康风险智能评估系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 高级UI样式 (CSS)
st.markdown("""
<style>
    /* 全局字体与背景 */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* 标题样式 */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* 侧边栏优化 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    
    /* 卡片样式 - 用于包裹结果 */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    /* 风险提示框 */
    .risk-alert-high {
        padding: 20px;
        border-radius: 8px;
        background-color: #fff5f5;
        border-left: 5px solid #fc8181;
        color: #c53030;
    }
    
    .risk-alert-low {
        padding: 20px;
        border-radius: 8px;
        background-color: #f0fff4;
        border-left: 5px solid #68d391;
        color: #276749;
    }
    
    /* 按钮样式优化 */
    div.stButton > button:first-child {
        background-color: #3182ce;
        color: white;
        border-radius: 6px;
        border: none;
        height: 50px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #2b6cb0;
        box-shadow: 0 4px 12px rgba(49, 130, 206, 0.3);
    }
    
    /* 隐藏Streamlit默认菜单 */
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
        st.error("未找到模型文件 'CatBoost.pkl'，请确保文件在当前目录下。")
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
    
    # ... (此处保留原本完整的if-else逻辑，为节省篇幅省略，请务必保留你原代码中的完整逻辑) ...
    # 为了代码简洁，我这里直接复用你原来提供的逻辑，请确保这里粘贴了你完整的 BMI 判断代码
    # -------------------------------------------------------------------------
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

# 6. 主界面布局

# 标题区域
st.title("学生肥胖风险评估系统")
st.markdown("基于机器学习的青少年健康风险预测模型")
st.markdown("---")

# 7. 侧边栏：使用 Form 表单（性能优化的关键！）
with st.sidebar:
    st.header("数据录入")
    # 使用 st.form 包裹所有输入项，这样只有点击提交按钮时才会刷新页面
    with st.form(key='prediction_form'):
        
        # 分组1：基本信息
        st.subheader("基本指标")
        col_a, col_b = st.columns(2)
        with col_a:
            GENDER = st.selectbox("性别", options=list(GENDER_options.keys()), format_func=lambda x: GENDER_options[x])
            height_cm = st.number_input("身高 (cm)", 100.0, 200.0, 150.0, 0.1)
        with col_b:
            AGE = st.selectbox("年龄", options=range(6, 19), format_func=lambda x: f"{x}岁")
            weight_kg = st.number_input("体重 (kg)", 20.0, 150.0, 45.0, 0.1)

        # 分组2：生活习惯
        st.subheader("饮食与运动")
        PEC = st.selectbox("每周体育课", list(PEC_options.keys()), format_func=lambda x: PEC_options[x])
        FrFF = st.selectbox("七天内新鲜水果", list(FrFF_options.keys()), format_func=lambda x: FrFF_options[x])
        DVT = st.selectbox("每天蔬菜种类", list(DVT_options.keys()), format_func=lambda x: DVT_options[x])
        HU = st.selectbox("耳机使用(>30分钟)", list(HU_options.keys()), format_func=lambda x: HU_options[x])

        # 分组3：心理与行为
        with st.expander("心理状态评估 (点击展开)", expanded=False):
            D1 = st.selectbox("小事烦恼", list(D1_options.keys()), format_func=lambda x: D1_options[x])
            D2 = st.selectbox("食欲不振", list(D2_options.keys()), format_func=lambda x: D2_options[x])
            D3 = st.selectbox("无法摆脱苦闷", list(D3_options.keys()), format_func=lambda x: D3_options[x])
            D9 = st.selectbox("觉得生活无用", list(D9_options.keys()), format_func=lambda x: D9_options[x])
            D11 = st.selectbox("睡眠无法解乏", list(D11_options.keys()), format_func=lambda x: D11_options[x])
            D17 = st.selectbox("曾经痛哭", list(D17_options.keys()), format_func=lambda x: D17_options[x])
        
        with st.expander("行为调查 (点击展开)", expanded=False):
            FF = st.selectbox("过去12个月打架", list(FF_options.keys()), format_func=lambda x: FF_options[x])
            PPP = st.selectbox("过去30天被责罚", list(PPP_options.keys()), format_func=lambda x: PPP_options[x])

        # 提交按钮
        submit_button = st.form_submit_button(label='开始分析预测')

# 8. 预测与结果显示区域
if submit_button:
    if model is not None:
        # 计算中间变量
        baseline_obesity = calculate_baseline_obesity(AGE, GENDER, height_cm, weight_kg)
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        obesity_status_text = "超重/肥胖" if baseline_obesity == 1 else "正常范围"

        # 准备特征
        feature_values = [GENDER, baseline_obesity, D2, AGE, D1, D9, HU, D11, PEC, FrFF, D17, DVT, FF, D3, PPP]
        features = np.array([feature_values])
        
        # 预测
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]
        risk_probability = predicted_proba[1]

        # --- 结果展示区 ---
        
        # 第一行：关键指标卡片
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #6c757d; font-size: 14px;">当前 BMI 指数</div>
                <div style="font-size: 32px; font-weight: bold; color: #2c3e50;">{bmi:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            color = "#e03131" if baseline_obesity == 1 else "#2f9e44"
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #6c757d; font-size: 14px;">当前体重状态</div>
                <div style="font-size: 24px; font-weight: bold; color: {color}; line-height: 1.5;">{obesity_status_text}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            # 动态计算风险颜色
            risk_color = "#e03131" if risk_probability > 0.5 else "#2f9e44"
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #6c757d; font-size: 14px;">预测肥胖风险概率</div>
                <div style="font-size: 32px; font-weight: bold; color: {risk_color};">{(risk_probability * 100):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("") # 占位符

        # 第二行：详细分析与图表
        col_main, col_viz = st.columns([1.5, 1])

        with col_main:
            st.subheader("分析报告")
            if predicted_class == 1:
                st.markdown(f"""
                <div class="risk-alert-high">
                    <h4>⚠️ 高风险预警</h4>
                    <p>根据模型分析，该学生在未来一年内面临较高的肥胖风险。建议立即采取干预措施。</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("##### 🩺 建议干预方案")
                st.markdown("""
                * **运动干预**：将每周体育课外的中等强度运动增加至每天60分钟。
                * **饮食调整**：减少高热量零食，增加蔬菜摄入（目前摄入量：{}）。
                * **心理支持**：关注情绪波动，目前的心理问卷显示可能存在压力源。
                """.format(DVT_options[DVT]))
            else:
                st.markdown(f"""
                <div class="risk-alert-low">
                    <h4>✅ 低风险状态</h4>
                    <p>根据模型分析，该学生目前的各项指标较为健康，未来肥胖风险较低。</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("##### 🌟 保持建议")
                st.markdown("""
                * 保持当前的运动频率。
                * 继续维持均衡的饮食结构。
                * 定期监测身高体重变化即可。
                """)

        with col_viz:
            st.subheader("概率分布可视化")
            
            # 使用 Altair 绘制更高级的图表 (替代 Matplotlib)
            chart_data = pd.DataFrame({
                '状态': ['低风险', '高风险'],
                '概率': [predicted_proba[0], predicted_proba[1]],
                'Color': ['#69db7c', '#ff8787']
            })

            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('概率', axis=alt.Axis(format='%'), title=None),
                y=alt.Y('状态', sort=None, title=None),
                color=alt.Color('Color', scale=None),
                tooltip=['状态', alt.Tooltip('概率', format='.1%')]
            ).properties(
                height=200
            ).configure_axis(
                labelFontSize=12,
                titleFontSize=14
            )
            
            st.altair_chart(chart, use_container_width=True)

    else:
        st.warning("请检查模型文件是否正确加载。")
else:
    # 初始状态，显示引导信息
    st.info("👈 请在左侧填写学生详细信息，并点击“开始分析预测”按钮获取结果。")
