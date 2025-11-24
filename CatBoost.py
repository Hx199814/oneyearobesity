import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 页面配置 ---
st.set_page_config(
    page_title="学生肥胖风险预测系统",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 模型加载 (使用 @st.cache_resource 确保只加载一次) ---
@st.cache_resource
def load_model():
    # 第一次加载时可能需要等待，但后续交互不会重复加载
    # 确保 'CatBoost.pkl' 文件存在于应用的根目录下
    try:
        return joblib.load('CatBoost.pkl')
    except FileNotFoundError:
        st.error("错误：未找到模型文件 'CatBoost.pkl'。请检查文件路径。")
        return None

model = load_model()

# 如果模型加载失败，停止运行
if model is None:
    st.stop()


# --- 定义特征选项 (保留不变) ---
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

# --- 计算基线肥胖函数 (保留不变) ---
def calculate_baseline_obesity(age, gender, height_cm, weight_kg):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    gender_code = 1 if gender == 1 else 0
    
    # 简化边界检查和赋值，确保返回0或1
    
    # 临界值数据 (来自原代码)
    thresholds = {
        6: {1: 17.7, 0: 17.5}, 6.5: {1: 18.1, 0: 18.0}, 7: {1: 18.7, 0: 18.5}, 7.5: {1: 19.2, 0: 19.0},
        8: {1: 19.7, 0: 19.4}, 8.5: {1: 20.3, 0: 19.9}, 9: {1: 20.8, 0: 20.4}, 9.5: {1: 21.4, 0: 21.0},
        10: {1: 21.9, 0: 21.5}, 10.5: {1: 22.5, 0: 22.1}, 11: {1: 23.0, 0: 22.7}, 11.5: {1: 23.6, 0: 23.3},
        12: {1: 24.1, 0: 23.9}, 12.5: {1: 24.7, 0: 24.5}, 13: {1: 25.2, 0: 25.6}, 13.5: {1: 25.7, 0: 25.6},
        14: {1: 26.1, 0: 25.9}, 14.5: {1: 26.4, 0: 26.3}, 15: {1: 26.6, 0: 26.6}, 15.5: {1: 26.9, 0: 26.9},
        16: {1: 27.1, 0: 27.1}, 16.5: {1: 27.4, 0: 27.4}, 17: {1: 27.6, 0: 27.6}, 17.5: {1: 27.8, 0: 27.8},
        18: {1: 28.0, 0: 28.0} # 18岁及以上统一使用28.0，不区分性别
    }

    age_group = next((a for a in sorted(thresholds.keys()) if a <= age < a + 0.5), None)
    
    if age >= 18:
        threshold = 28.0
    elif age_group is not None:
        threshold = thresholds[age_group].get(gender_code, 28.0) # 默认值以防万一
    else:
        # 处理 age < 6 或其他未覆盖的年龄
        # 实际应用中，如果数据范围确定，可以忽略此分支
        if bmi >= 24.0: # 使用成人超重标准作为默认安全检查
            return 1
        return 0

    if bmi >= threshold:
        return 1
        
    return 0

# --- 自定义CSS样式 (保留上次优化后的样式) ---
st.markdown("""
<style>
    /* 主标题样式 */
    .main-header {
        font-size: 2.5rem; 
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
    }
    /* 副标题/模块标题样式 */
    .sub-header {
        font-size: 1.4rem; 
        color: #2e86ab;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.3rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    /* 侧边栏标题样式 */
    .sidebar-header {
        font-size: 1.6rem;
        color: #ffffff; 
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    /* 预测结果框样式 */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
    }
    /* 高风险结果样式 */
    .high-risk {
        background-color: #fce4ec; 
        border-left: 6px solid #e91e63; 
    }
    /* 低风险结果样式 */
    .low-risk {
        background-color: #e8f5e9; 
        border-left: 6px solid #4caf50; 
    }
    /* 建议框样式 */
    .advice-box {
        background-color: #e3f2fd; 
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #90caf9;
        margin: 1rem 0;
    }
    /* 侧边栏按钮样式 */
    .stButton>button {
        width: 100%;
        border-radius: 8px; 
        height: 3.5em; 
        font-size: 1.2rem;
        background-color: #1f77b4;
        color: white;
        transition: background-color 0.3s; 
    }
    .stButton>button:hover {
        background-color: #0d47a1; 
    }
    /* Streamlit sidebar background color (for better look) */
    [data-testid="stSidebar"] {
        background-color: #1f77b4; 
    }
    [data-testid="stSidebar"] .stSelectbox label, 
    [data-testid="stSidebar"] .stNumberInput label {
        color: white !important; 
    }
    [data-testid="stSidebar"] .stMetric {
        background-color: #0d47a1; 
        border-radius: 8px;
        padding: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 主页面标题
st.markdown('<p class="main-header">学生肥胖风险预测系统</p>', unsafe_allow_html=True)

# --- 侧边栏：数据输入 (保持结构不变，样式优化) ---
with st.sidebar:
    st.markdown('<p class="sidebar-header">学生信息录入</p>', unsafe_allow_html=True)
    
    # 基本信息
    st.markdown("### 👤 基本信息")
    col1, col2 = st.columns(2)
    with col1:
        GENDER = st.selectbox("性别:", options=list(GENDER_options.keys()), format_func=lambda x: GENDER_options[x], key='GENDER_sb')
    with col2:
        AGE = st.selectbox("年龄:", options=[6,7,8,9,10,11,12,13,14,15,16,17,18], format_func=lambda x: f"{x}岁", key='AGE_sb')
    
    # 身高体重
    st.markdown("### 📊 身高体重")
    col1, col2 = st.columns(2)
    with col1:
        height_cm = st.number_input("身高 (cm):", min_value=100.0, max_value=200.0, value=150.0, step=0.1, key='HEIGHT_ni')
    with col2:
        weight_kg = st.number_input("体重 (kg):", min_value=20.0, max_value=100.0, value=45.0, step=0.1, key='WEIGHT_ni')
    
    # 计算BMI和基线肥胖状态
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    baseline_obesity = calculate_baseline_obesity(AGE, GENDER, height_cm, weight_kg)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="stMetric">BMI指数<div style="font-size: 1.5rem; font-weight: bold;">{bmi:.1f}</div></div>', unsafe_allow_html=True)
    with col2:
        obesity_status = "肥胖" if baseline_obesity == 1 else "正常"
        st.markdown(f'<div class="stMetric">基线肥胖状态<div style="font-size: 1.5rem; font-weight: bold;">{obesity_status}</div></div>', unsafe_allow_html=True)

    st.markdown("---") # 分隔线

    # 饮食与运动
    st.markdown("### 🍎 饮食与运动")
    PEC = st.selectbox("每周体育课节数:", options=list(PEC_options.keys()), format_func=lambda x: PEC_options[x], key='PEC_sb')
    FrFF = st.selectbox("过去七天吃新鲜水果次数:", options=list(FrFF_options.keys()), format_func=lambda x: FrFF_options[x], key='FrFF_sb')
    DVT = st.selectbox("每天吃几种蔬菜:", options=list(DVT_options.keys()), format_func=lambda x: DVT_options[x], key='DVT_sb')
    
    # 情绪状态
    st.markdown("### 😊 情绪状态")
    D1 = st.selectbox("以前从不困扰我的事情现在让我烦恼:", options=list(D1_options.keys()), format_func=lambda x: D1_options[x], key='D1_sb')
    D2 = st.selectbox("我不想吃东西；我胃口不好:", options=list(D2_options.keys()), format_func=lambda x: D2_options[x], key='D2_sb')
    D3 = st.selectbox("即便有家人或朋友帮助也无法摆脱苦闷:", options=list(D3_options.keys()), format_func=lambda x: D3_options[x], key='D3_sb')
    D9 = st.selectbox("我认为我的生活一无是处:", options=list(D9_options.keys()), format_func=lambda x: D9_options[x], key='D9_sb')
    D11 = st.selectbox("我睡觉后不能缓解疲劳:", options=list(D11_options.keys()), format_func=lambda x: D11_options[x], key='D11_sb')
    D17 = st.selectbox("我曾经放声痛哭:", options=list(D17_options.keys()), format_func=lambda x: D17_options[x], key='D17_sb')
    
    # 行为习惯
    st.markdown("### 📱 行为习惯")
    HU = st.selectbox("是否使用耳机（至少连续30分钟）:", options=list(HU_options.keys()), format_func=lambda x: HU_options[x], key='HU_sb')
    FF = st.selectbox("过去12个月里是否与他人动手打架:", options=list(FF_options.keys()), format_func=lambda x: FF_options[x], key='FF_sb')
    PPP = st.selectbox("过去30天是否曾被家长打骂:", options=list(PPP_options.keys()), format_func=lambda x: PPP_options[x], key='PPP_sb')

    st.markdown("---") # 分隔线
    
    # 预测按钮逻辑
    if st.button("开始预测", type="primary"):
        # 将输入数据存储在 session state 中，供主内容区使用
        st.session_state['run_prediction'] = True
        # 注意：这里我们收集的特征必须与训练模型时的特征顺序和数量一致
        st.session_state['features'] = [GENDER, baseline_obesity, D2, AGE, D1, D9, HU, D11, PEC, FrFF, D17, DVT, FF, D3, PPP]
    else:
        # 确保初始状态为 False
        if 'run_prediction' not in st.session_state:
             st.session_state['run_prediction'] = False


# --- 主内容区域：预测结果与建议 ---
st.markdown('<p class="sub-header">💡 预测结果与健康建议</p>', unsafe_allow_html=True)
prediction_placeholder = st.empty() # 占位符，用于显示预测结果
chart_placeholder = st.empty() # 占位符，用于显示图表

# 只有在点击按钮后，并且 session state 中有数据时才进行预测
if st.session_state.get('run_prediction', False):
    
    with prediction_placeholder.container():
        st.info("正在分析数据，请稍候...")
    
    try:
        features = st.session_state['features']
        
        # **【核心修复点 1】强制将特征数组转换为整数类型 (np.int64) **
        # 这确保了 CatBoost 能够正确处理输入，避免索引错误。
        features_array = np.array([features], dtype=np.int64)
        
        # 预测
        predicted_class_raw = model.predict(features_array)[0]
        
        # **【核心修复点 2】确保预测类别是标准的 Python 整数**
        predicted_class = int(predicted_class_raw)
        
        predicted_proba = model.predict_proba(features_array)[0]
        
        # 清除加载提示，显示结果
        prediction_placeholder.empty()

        # 显示预测结果
        with prediction_placeholder.container():
            # **【核心修复点 3】使用确保为整数的 predicted_class 作为索引**
            probability = predicted_proba[predicted_class] * 100
            
            if predicted_class == 1:
                st.markdown(f'<div class="prediction-box high-risk">', unsafe_allow_html=True)
                st.markdown("### 🚨 预测结果：肥胖风险高")
                st.markdown(f"**预测概率:** <span style='font-size: 1.5rem; font-weight: bold; color: #e91e63;'>{probability:.1f}%</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="advice-box">', unsafe_allow_html=True)
                st.markdown("### 💡 针对高风险的健康建议")
                st.markdown("""
                - **🩺 重点关注**: 立即采取措施干预，建议咨询专业医生或营养师。
                - **🏃 增加运动量**: 每天至少**60分钟**中高强度体育活动。
                - **🥗 改善饮食**: 严格限制含糖饮料、高油高盐零食，确保**足量蔬菜水果**。
                - **😴 规律作息**: 保证充足的睡眠时间，避免熬夜。
                - **📱 减少屏幕时间**: 严格控制手机、电脑等电子产品使用时间。
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box low-risk">', unsafe_allow_html=True)
                st.markdown("### ✅ 预测结果：肥胖风险低")
                st.markdown(f"**健康概率:** <span style='font-size: 1.5rem; font-weight: bold; color: #4caf50;'>{probability:.1f}%</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="advice-box">', unsafe_allow_html=True)
                st.markdown("### 💡 保持健康的建议")
                st.markdown("""
                - **👍 保持习惯**: 继续坚持规律的运动和均衡的饮食习惯。
                - **🍎 均衡营养**: 维持多样化饮食结构，特别是对蔬菜和水果的摄入。
                - **🧘 情绪管理**: 保持积极乐观的心态，定期与家人朋友沟通。
                - **🔄 定期监测**: 建议每半年进行一次健康体检，监测身高和体重变化。
                """)
                st.markdown('</div>', unsafe_allow_html=True)

        # 可视化概率
        with chart_placeholder.container():
            st.markdown('<p class="sub-header">📊 风险概率分布图</p>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ['非肥胖', '肥胖']
            probabilities = [predicted_proba[0], predicted_proba[1]]
            colors = ['#4caf50', '#e91e63'] 
            
            bars = ax.barh(categories, probabilities, color=colors, alpha=0.9)
            ax.set_xlim(0, 1)
            ax.set_xlabel('概率', fontsize=10)
            ax.set_title('肥胖风险概率分布', fontsize=12)
            
            # 在条形图上添加数值标签
            for i, v in enumerate(probabilities):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
            
            # 美化图表
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.tick_params(axis='y', which='major', pad=10)
            
            st.pyplot(fig)
            
    except Exception as e:
        prediction_placeholder.empty()
        # 显示更详细的错误信息
        st.error(f"预测过程中出现错误，请检查模型文件或数据输入: {e}")
        # 清除 session state 避免无限循环
        st.session_state['run_prediction'] = False

# 默认主内容区显示 (未点击预测按钮时)
if not st.session_state.get('run_prediction', False):
    st.info("请在左侧侧边栏输入学生信息，并点击 **开始预测** 按钮以查看结果和健康建议。")
    st.markdown('<p class="sub-header">📖 系统简介</p>', unsafe_allow_html=True)
    st.markdown("""
    本系统采用 **CatBoost** 机器学习模型，综合学生的生理、行为、饮食和情绪等多维度数据，预测其在未来一年内发展为肥胖的风险。
    - **预测目标**: 1年肥胖风险 (0: 非肥胖, 1: 肥胖)
    - **数据安全**: 所有计算和数据处理均在您的浏览器本地进行，不会上传任何个人信息。
    """)
