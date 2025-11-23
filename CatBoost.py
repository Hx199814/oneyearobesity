import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 页面配置
st.set_page_config(
    page_title="学生肥胖风险预测系统",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load('CatBoost.pkl')

model = load_model()

# 定义特征选项
GENDER_options = {
    1: '男生', 
    2: '女生'
}

D2_options = {
    1: '没有或偶尔',  
    2: '有时',  
    3: '时常或一半时间',
    4: '多数时间或持续',
    5: '不清楚'
}

D1_options = {
    1: '没有或偶尔',  
    2: '有时',  
    3: '时常或一半时间',
    4: '多数时间或持续',
    5: '不清楚'
}

D9_options = {
    1: '没有或偶尔',  
    2: '有时',  
    3: '时常或一半时间',
    4: '多数时间或持续',
    5: '不清楚'
}

HU_options = {
    1: '不会',  
    2: '会'
}

D11_options = {
    1: '没有或偶尔',  
    2: '有时',  
    3: '时常或一半时间',
    4: '多数时间或持续',
    5: '不清楚'
}

PEC_options = {
    1: '0节',  
    2: '1节',  
    3: '2节',
    4: '3节',
    5: '4节',  
    6: '5节及以上'
}

FrFF_options = {
    1: '从来不吃',  
    2: '少于每天1次',  
    3: '每天1次',
    4: '每天2次及以上'
}

D17_options = {
    1: '没有或偶尔',  
    2: '有时',  
    3: '时常或一半时间',
    4: '多数时间或持续',
    5: '不清楚'
}

DVT_options = {
    1: '从来不吃或少于每天1种',  
    2: '每天1种',  
    3: '每天2种',
    4: '每天3次及以上'
}

FF_options = {
    1: '是',  
    0: '否'
}

D3_options = {
    1: '没有或偶尔',  
    2: '有时',  
    3: '时常或一半时间',
    4: '多数时间或持续',
    5: '不清楚'
}

PPP_options = {
    1: '是',  
    0: '否'
}

# 计算基线肥胖函数
def calculate_baseline_obesity(age, gender, height_cm, weight_kg):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    gender_code = 1 if gender == 1 else 0
    
    if age >= 6 and age < 6.5:
        if gender_code == 1 and bmi >= 17.7:
            return 1
        elif gender_code == 0 and bmi >= 17.5:
            return 1
    elif age >= 6.5 and age < 7:
        if gender_code == 1 and bmi >= 18.1:
            return 1
        elif gender_code == 0 and bmi >= 18.0:
            return 1
    elif age >= 7 and age < 7.5:
        if gender_code == 1 and bmi >= 18.7:
            return 1
        elif gender_code == 0 and bmi >= 18.5:
            return 1
    elif age >= 7.5 and age < 8:
        if gender_code == 1 and bmi >= 19.2:
            return 1
        elif gender_code == 0 and bmi >= 19.0:
            return 1
    elif age >= 8 and age < 8.5:
        if gender_code == 1 and bmi >= 19.7:
            return 1
        elif gender_code == 0 and bmi >= 19.4:
            return 1
    elif age >= 8.5 and age < 9:
        if gender_code == 1 and bmi >= 20.3:
            return 1
        elif gender_code == 0 and bmi >= 19.9:
            return 1
    elif age >= 9 and age < 9.5:
        if gender_code == 1 and bmi >= 20.8:
            return 1
        elif gender_code == 0 and bmi >= 20.4:
            return 1
    elif age >= 9.5 and age < 10:
        if gender_code == 1 and bmi >= 21.4:
            return 1
        elif gender_code == 0 and bmi >= 21.0:
            return 1
    elif age >= 10 and age < 10.5:
        if gender_code == 1 and bmi >= 21.9:
            return 1
        elif gender_code == 0 and bmi >= 21.5:
            return 1
    elif age >= 10.5 and age < 11:
        if gender_code == 1 and bmi >= 22.5:
            return 1
        elif gender_code == 0 and bmi >= 22.1:
            return 1
    elif age >= 11 and age < 11.5:
        if gender_code == 1 and bmi >= 23.0:
            return 1
        elif gender_code == 0 and bmi >= 22.7:
            return 1
    elif age >= 11.5 and age < 12:
        if gender_code == 1 and bmi >= 23.6:
            return 1
        elif gender_code == 0 and bmi >= 23.3:
            return 1
    elif age >= 12 and age < 12.5:
        if gender_code == 1 and bmi >= 24.1:
            return 1
        elif gender_code == 0 and bmi >= 23.9:
            return 1
    elif age >= 12.5 and age < 13:
        if gender_code == 1 and bmi >= 24.7:
            return 1
        elif gender_code == 0 and bmi >= 24.5:
            return 1
    elif age >= 13 and age < 13.5:
        if gender_code == 1 and bmi >= 25.2:
            return 1
        elif gender_code == 0 and bmi >= 25.6:
            return 1
    elif age >= 13.5 and age < 14:
        if gender_code == 1 and bmi >= 25.7:
            return 1
        elif gender_code == 0 and bmi >= 25.6:
            return 1
    elif age >= 14 and age < 14.5:
        if gender_code == 1 and bmi >= 26.1:
            return 1
        elif gender_code == 0 and bmi >= 25.9:
            return 1
    elif age >= 14.5 and age < 15:
        if gender_code == 1 and bmi >= 26.4:
            return 1
        elif gender_code == 0 and bmi >= 26.3:
            return 1
    elif age >= 15 and age < 15.5:
        if gender_code == 1 and bmi >= 26.6:
            return 1
        elif gender_code == 0 and bmi >= 26.6:
            return 1
    elif age >= 15.5 and age < 16:
        if gender_code == 1 and bmi >= 26.9:
            return 1
        elif gender_code == 0 and bmi >= 26.9:
            return 1
    elif age >= 16 and age < 16.5:
        if gender_code == 1 and bmi >= 27.1:
            return 1
        elif gender_code == 0 and bmi >= 27.1:
            return 1
    elif age >= 16.5 and age < 17:
        if gender_code == 1 and bmi >= 27.4:
            return 1
        elif gender_code == 0 and bmi >= 27.4:
            return 1
    elif age >= 17 and age < 17.5:
        if gender_code == 1 and bmi >= 27.6:
            return 1
        elif gender_code == 0 and bmi >= 27.6:
            return 1
    elif age >= 17.5 and age < 18:
        if gender_code == 1 and bmi >= 27.8:
            return 1
        elif gender_code == 0 and bmi >= 27.8:
            return 1
    elif age >= 18:
        if bmi >= 28.0:
            return 1
    
    return 0

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .sidebar-header {
        font-size: 1.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .low-risk {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .high-risk {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .advice-box {
        background-color: #e7f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-size: 1.2rem;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 主页面标题
st.markdown('<p class="main-header">🏫 学生肥胖风险预测系统</p>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown('<p class="sidebar-header">📝 学生信息录入</p>', unsafe_allow_html=True)
    
    # 基本信息
    st.markdown("### 👤 基本信息")
    col1, col2 = st.columns(2)
    with col1:
        GENDER = st.selectbox("性别:", options=list(GENDER_options.keys()), format_func=lambda x: GENDER_options[x])
    with col2:
        AGE = st.selectbox("年龄:", options=[6,7,8,9,10,11,12,13,14,15,16,17,18], format_func=lambda x: f"{x}岁")
    
    # 身高体重
    st.markdown("### 📊 身高体重")
    col1, col2 = st.columns(2)
    with col1:
        height_cm = st.number_input("身高 (cm):", min_value=100.0, max_value=200.0, value=150.0, step=0.1)
    with col2:
        weight_kg = st.number_input("体重 (kg):", min_value=20.0, max_value=100.0, value=45.0, step=0.1)
    
    # 计算基线肥胖
    baseline_obesity = calculate_baseline_obesity(AGE, GENDER, height_cm, weight_kg)
    
    # 显示BMI和基线肥胖状态
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("BMI指数", f"{bmi:.1f}")
    with col2:
        obesity_status = "肥胖" if baseline_obesity == 1 else "正常"
        st.metric("基线肥胖状态", obesity_status)
    
    # 饮食与运动
    st.markdown("### 🍎 饮食与运动")
    PEC = st.selectbox("每周体育课节数:", options=list(PEC_options.keys()), format_func=lambda x: PEC_options[x])
    FrFF = st.selectbox("过去七天吃新鲜水果次数:", options=list(FrFF_options.keys()), format_func=lambda x: FrFF_options[x])
    DVT = st.selectbox("每天吃几种蔬菜:", options=list(DVT_options.keys()), format_func=lambda x: DVT_options[x])
    
    # 情绪状态
    st.markdown("### 😊 情绪状态")
    D1 = st.selectbox("以前从不困扰我的事情现在让我烦恼:", options=list(D1_options.keys()), format_func=lambda x: D1_options[x])
    D2 = st.selectbox("我不想吃东西；我胃口不好:", options=list(D2_options.keys()), format_func=lambda x: D2_options[x])
    D3 = st.selectbox("我觉得即便有家人或朋友帮助也无法摆脱这种苦闷:", options=list(D3_options.keys()), format_func=lambda x: D3_options[x])
    D9 = st.selectbox("我认为我的生活一无是处:", options=list(D9_options.keys()), format_func=lambda x: D9_options[x])
    D11 = st.selectbox("我睡觉后不能缓解疲劳:", options=list(D11_options.keys()), format_func=lambda x: D11_options[x])
    D17 = st.selectbox("我曾经放声痛哭:", options=list(D17_options.keys()), format_func=lambda x: D17_options[x])
    
    # 行为习惯
    st.markdown("### 📱 行为习惯")
    HU = st.selectbox("是否使用耳机（至少连续30分钟）:", options=list(HU_options.keys()), format_func=lambda x: HU_options[x])
    FF = st.selectbox("过去12个月里是否与他人动手打架:", options=list(FF_options.keys()), format_func=lambda x: FF_options[x])
    PPP = st.selectbox("过去30天是否曾被家长打骂:", options=list(PPP_options.keys()), format_func=lambda x: PPP_options[x])

# 主内容区域
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<p class="sub-header">🔍 预测分析</p>', unsafe_allow_html=True)
    
    if st.button("开始预测", type="primary"):
        with st.spinner("正在分析数据，请稍候..."):
            try:
                # 准备特征数据
                feature_values = [GENDER, baseline_obesity, D2, AGE, D1, D9, HU, D11, PEC, FrFF, D17, DVT, FF, D3, PPP]
                features = np.array([feature_values])
                
                # 预测
                predicted_class = model.predict(features)[0] 
                predicted_proba = model.predict_proba(features)[0]
                
                # 显示预测结果
                probability = predicted_proba[predicted_class] * 100
                
                if predicted_class == 1:
                    st.markdown(f'<div class="prediction-box high-risk">', unsafe_allow_html=True)
                    st.markdown("### ⚠️ 肥胖风险高")
                    st.markdown(f"**风险概率:** {probability:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="advice-box">', unsafe_allow_html=True)
                    st.markdown("### 💡 健康建议")
                    st.markdown("""
                    - **增加体育锻炼**: 每天至少60分钟中等强度运动
                    - **改善饮食习惯**: 减少高糖高脂食物，增加蔬菜水果摄入
                    - **控制屏幕时间**: 每天不超过2小时
                    - **保证充足睡眠**: 每天9-11小时
                    - **定期健康检查**: 建议每半年测量一次身高体重
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box low-risk">', unsafe_allow_html=True)
                    st.markdown("### ✅ 肥胖风险低")
                    st.markdown(f"**健康概率:** {probability:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="advice-box">', unsafe_allow_html=True)
                    st.markdown("### 💡 保持建议")
                    st.markdown("""
                    - **保持运动习惯**: 继续坚持规律锻炼
                    - **均衡饮食**: 维持多样化饮食结构
                    - **规律作息**: 保持充足睡眠和规律生活
                    - **定期监测**: 建议每年测量一次身高体重
                    - **情绪状态**: 保持积极乐观心态
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 可视化概率
                st.markdown('<p class="sub-header">📈 风险概率分布</p>', unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                categories = ['非肥胖', '肥胖']
                probabilities = [predicted_proba[0], predicted_proba[1]]
                colors = ['#28a745', '#dc3545']
                
                bars = ax.barh(categories, probabilities, color=colors, alpha=0.8)
                ax.set_xlim(0, 1)
                ax.set_xlabel('概率', fontsize=12, fontweight='bold')
                ax.set_title('肥胖风险概率分布', fontsize=14, fontweight='bold')
                
                # 在条形图上添加数值标签
                for i, v in enumerate(probabilities):
                    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=11, fontweight='bold')
                
                # 美化图表
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(axis='y', which='major', pad=15)
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"预测过程中出现错误: {str(e)}")

with col2:
    st.markdown('<p class="sub-header">ℹ️ 系统说明</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px;">
    <h4>关于本系统</h4>
    <p>本系统基于机器学习算法，综合分析学生的生理指标、饮食习惯、运动情况和情绪状态，预测1年后肥胖风险。</p>
    
    <h4>使用说明</h4>
    <ol>
    <li>在左侧填写学生信息</li>
    <li>点击"开始预测"按钮</li>
    <li>查看预测结果和建议</li>
    </ol>
    
    <h4>数据安全</h4>
    <p>所有数据仅在本地处理，不会上传到服务器，保障隐私安全。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 健康小贴士
    st.markdown('<p class="sub-header">💪 健康小贴士</p>', unsafe_allow_html=True)
    
    tips = [
        "每天保证1小时户外活动",
        "饮食多样化，多吃蔬菜水果",
        "限制油炸食品和零食",
        "保持积极乐观心态",
        "定期进行健康检查"
    ]
    
    for i, tip in enumerate(tips):
        st.markdown(f"""
        <div class="metric-box">
        <span style="font-size: 1.2rem;">{tip}</span>
        </div>
        """, unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d;'>"
    "学生肥胖风险预测系统 © 2024 | 健康管理专家"
    "</div>",
    unsafe_allow_html=True
)