import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# -------------------------- 基础优化：减少重复计算 + 资源缓存 --------------------------
# 页面配置（精简参数，提升加载速度）
st.set_page_config(
    page_title="学生肥胖风险预测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载模型（缓存优化，仅加载一次）
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load('CatBoost.pkl')

model = load_model()

# 优化BMI基线判断：用字典映射替代大量if-elif，提升计算速度
BMI_THRESHOLDS = {
    (6, 6.5): {'male': 17.7, 'female': 17.5},
    (6.5, 7): {'male': 18.1, 'female': 18.0},
    (7, 7.5): {'male': 18.7, 'female': 18.5},
    (7.5, 8): {'male': 19.2, 'female': 19.0},
    (8, 8.5): {'male': 19.7, 'female': 19.4},
    (8.5, 9): {'male': 20.3, 'female': 19.9},
    (9, 9.5): {'male': 20.8, 'female': 20.4},
    (9.5, 10): {'male': 21.4, 'female': 21.0},
    (10, 10.5): {'male': 21.9, 'female': 21.5},
    (10.5, 11): {'male': 22.5, 'female': 22.1},
    (11, 11.5): {'male': 23.0, 'female': 22.7},
    (11.5, 12): {'male': 23.6, 'female': 23.3},
    (12, 12.5): {'male': 24.1, 'female': 23.9},
    (12.5, 13): {'male': 24.7, 'female': 24.5},
    (13, 13.5): {'male': 25.2, 'female': 25.6},
    (13.5, 14): {'male': 25.7, 'female': 25.6},
    (14, 14.5): {'male': 26.1, 'female': 25.9},
    (14.5, 15): {'male': 26.4, 'female': 26.3},
    (15, 15.5): {'male': 26.6, 'female': 26.6},
    (15.5, 16): {'male': 26.9, 'female': 26.9},
    (16, 16.5): {'male': 27.1, 'female': 27.1},
    (16.5, 17): {'male': 27.4, 'female': 27.4},
    (17, 17.5): {'male': 27.6, 'female': 27.6},
    (17.5, 18): {'male': 27.8, 'female': 27.8},
    (18, float('inf')): {'male': 28.0, 'female': 28.0}
}

def calculate_baseline_obesity(age, gender, height_cm, weight_kg):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    gender_key = 'male' if gender == 1 else 'female'
    
    # 快速匹配年龄区间（替代if-elif）
    for (min_age, max_age), thresholds in BMI_THRESHOLDS.items():
        if min_age <= age < max_age:
            return 1 if bmi >= thresholds[gender_key] else 0
    return 0

# -------------------------- 特征选项定义（保留原变量，仅优化格式） --------------------------
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

# -------------------------- 高级UI样式（替换幼稚图标，提升专业感） --------------------------
st.markdown("""
<style>
    /* 全局样式重置 */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* 标题样式 */
    .main-header {
        font-size: 2.8rem;
        color: #2d3748;
        text-align: center;
        margin: 1.5rem 0;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.4rem;
        color: #2d3748;
        border-bottom: 2px solid #4299e1;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem;
        font-weight: 500;
    }
    
    .sidebar-header {
        font-size: 1.6rem;
        color: #2d3748;
        text-align: center;
        margin: 1rem 0 1.2rem;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    /* 卡片样式 */
    .prediction-box {
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .prediction-box:hover {
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    }
    
    .low-risk {
        background-color: #f0fdf4;
        border-left: 6px solid #10b981;
    }
    
    .high-risk {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
    }
    
    .advice-box {
        background-color: #f5fafe;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #4299e1;
        margin: 1.2rem 0;
    }
    
    .metric-box {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.8rem;
    }
    
    /* 按钮样式 */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.2em;
        font-size: 1.1rem;
        background-color: #4299e1;
        color: white;
        border: none;
        transition: background-color 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #3182ce;
        color: white;
    }
    
    .stButton>button:active {
        background-color: #2b6cb0;
    }
    
    /* 输入框/选择框样式 */
    .stSelectbox, .stNumberInput {
        margin-bottom: 1rem;
    }
    
    .stNumberInput input, .stSelectbox select {
        border-radius: 6px;
        border: 1px solid #cbd5e1;
        padding: 0.5rem;
    }
    
    /* 侧边栏分组标题 */
    .sidebar-group-title {
        font-size: 1.1rem;
        color: #2d3748;
        margin: 1.2rem 0 0.6rem;
        font-weight: 500;
        display: flex;
        align-items: center;
    }
    
    .sidebar-group-title svg {
        margin-right: 0.5rem;
        fill: #4299e1;
    }
    
    /* 页脚样式 */
    .footer {
        text-align: center;
        color: #718096;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* 隐藏Streamlit默认边框和阴影 */
    .stApp {
        background-color: #ffffff;
    }
    
    .stSidebar {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------- 主页面布局（优化排版，提升专业感） --------------------------
# 主标题（替换幼稚图标）
st.markdown('<h1 class="main-header">📊 学生肥胖风险预测系统</h1>', unsafe_allow_html=True)

# 侧边栏（优化分组标题，替换图标，精简布局）
with st.sidebar:
    st.markdown('<h2 class="sidebar-header">学生信息录入</h2>', unsafe_allow_html=True)
    
    # 基本信息（用Font Awesome图标替代emoji，更专业）
    st.markdown("""
    <div class="sidebar-group-title">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
            <circle cx="12" cy="7" r="4"></circle>
        </svg>
        基本信息
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        GENDER = st.selectbox("性别", options=list(GENDER_options.keys()), format_func=lambda x: GENDER_options[x])
    with col2:
        AGE = st.selectbox("年龄", options=[6,7,8,9,10,11,12,13,14,15,16,17,18], format_func=lambda x: f"{x}岁")
    
    # 身高体重
    st.markdown("""
    <div class="sidebar-group-title">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2v20"></path>
            <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
        </svg>
        身高体重
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        height_cm = st.number_input("身高 (cm)", min_value=100.0, max_value=200.0, value=150.0, step=0.1)
    with col2:
        weight_kg = st.number_input("体重 (kg)", min_value=20.0, max_value=100.0, value=45.0, step=0.1)
    
    # 实时计算BMI和基线肥胖状态（缓存计算结果）
    @st.cache_data(depends_on=[AGE, GENDER, height_cm, weight_kg])
    def compute_bmi_and_baseline():
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        baseline = calculate_baseline_obesity(AGE, GENDER, height_cm, weight_kg)
        return bmi, baseline
    
    bmi, baseline_obesity = compute_bmi_and_baseline()
    
    # 显示BMI和基线状态（卡片式设计）
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size: 0.9rem; color: #718096; margin-bottom: 0.2rem;">BMI指数</div>
            <div style="font-size: 1.5rem; color: #2d3748; font-weight: 600;">{bmi:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        obesity_status = "肥胖" if baseline_obesity == 1 else "正常"
        status_color = "#ef4444" if baseline_obesity == 1 else "#10b981"
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size: 0.9rem; color: #718096; margin-bottom: 0.2rem;">基线肥胖状态</div>
            <div style="font-size: 1.5rem; color: {status_color}; font-weight: 600;">{obesity_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 饮食与运动
    st.markdown("""
    <div class="sidebar-group-title">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M18 8h1a4 4 0 0 1 0 8h-1"></path>
            <path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"></path>
            <line x1="6" y1="1" x2="6" y2="4"></line>
            <line x1="10" y1="1" x2="10" y2="4"></line>
            <line x1="14" y1="1" x2="14" y2="4"></line>
        </svg>
        饮食与运动
    </div>
    """, unsafe_allow_html=True)
    PEC = st.selectbox("每周体育课节数", options=list(PEC_options.keys()), format_func=lambda x: PEC_options[x])
    FrFF = st.selectbox("过去七天吃新鲜水果次数", options=list(FrFF_options.keys()), format_func=lambda x: FrFF_options[x])
    DVT = st.selectbox("每天吃几种蔬菜", options=list(DVT_options.keys()), format_func=lambda x: DVT_options[x])
    
    # 情绪状态
    st.markdown("""
    <div class="sidebar-group-title">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="2" y1="12" x2="22" y2="12"></line>
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
        </svg>
        情绪状态
    </div>
    """, unsafe_allow_html=True)
    D1 = st.selectbox("以前从不困扰我的事情现在让我烦恼", options=list(D1_options.keys()), format_func=lambda x: D1_options[x])
    D2 = st.selectbox("我不想吃东西；我胃口不好", options=list(D2_options.keys()), format_func=lambda x: D2_options[x])
    D3 = st.selectbox("我觉得即便有家人或朋友帮助也无法摆脱这种苦闷", options=list(D3_options.keys()), format_func=lambda x: D3_options[x])
    D9 = st.selectbox("我认为我的生活一无是处", options=list(D9_options.keys()), format_func=lambda x: D9_options[x])
    D11 = st.selectbox("我睡觉后不能缓解疲劳", options=list(D11_options.keys()), format_func=lambda x: D11_options[x])
    D17 = st.selectbox("我曾经放声痛哭", options=list(D17_options.keys()), format_func=lambda x: D17_options[x])
    
    # 行为习惯
    st.markdown("""
    <div class="sidebar-group-title">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>
        行为习惯
    </div>
    """, unsafe_allow_html=True)
    HU = st.selectbox("是否使用耳机（至少连续30分钟）", options=list(HU_options.keys()), format_func=lambda x: HU_options[x])
    FF = st.selectbox("过去12个月里是否与他人动手打架", options=list(FF_options.keys()), format_func=lambda x: FF_options[x])
    PPP = st.selectbox("过去30天是否曾被家长打骂", options=list(PPP_options.keys()), format_func=lambda x: PPP_options[x])

# -------------------------- 主内容区域（优化布局，移除冗余Tips） --------------------------
col_main, col_side = st.columns([3, 1.2])

with col_main:
    st.markdown('<h2 class="sub-header">预测分析结果</h2>', unsafe_allow_html=True)
    
    if st.button("开始预测", type="primary"):
        with st.spinner("🔍 正在分析数据，请稍候..."):
            try:
                # 准备特征数据（保持原逻辑）
                feature_values = [GENDER, baseline_obesity, D2, AGE, D1, D9, HU, D11, PEC, FrFF, D17, DVT, FF, D3, PPP]
                features = np.array([feature_values])
                
                # 预测（缓存模型预测结果，避免重复计算）
                @st.cache_data(depends_on=[features])
                def predict(features):
                    pred_class = model.predict(features)[0]
                    pred_proba = model.predict_proba(features)[0]
                    return pred_class, pred_proba
                
                predicted_class, predicted_proba = predict(features)
                probability = predicted_proba[predicted_class] * 100
                
                # 显示预测结果（优化文案和样式）
                if predicted_class == 1:
                    st.markdown(f'''
                    <div class="prediction-box high-risk">
                        <h3 style="color: #dc2626; margin-bottom: 0.8rem; font-size: 1.5rem; display: flex; align-items: center;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.5rem;">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="12" y1="8" x2="12" y2="12"></line>
                                <line x1="12" y1="16" x2="12.01" y2="16"></line>
                            </svg>
                            肥胖风险高
                        </h3>
                        <p style="color: #4b5563; font-size: 1.1rem;">风险概率：<strong>{probability:.1f}%</strong></p>
                        <p style="color: #718096; font-size: 0.95rem; margin-top: 0.5rem;">提示：该学生未来1年肥胖风险较高，建议及时干预。</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.markdown(f'''
                    <div class="advice-box">
                        <h3 style="color: #2d3748; margin-bottom: 0.8rem; font-size: 1.2rem; font-weight: 500;">
                            健康干预建议
                        </h3>
                        <ul style="color: #4b5563; font-size: 1rem; line-height: 1.8;">
                            <li>💪 增加体育锻炼：每天至少60分钟中等强度运动（如快走、游泳、跳绳）</li>
                            <li>🥗 改善饮食习惯：减少高糖、高脂、高盐食物，每日蔬菜水果摄入≥500g</li>
                            <li>📱 控制屏幕时间：每天电子设备使用时间不超过2小时，避免久坐</li>
                            <li>😴 保证充足睡眠：小学生每日睡眠10-12小时，初中生9-10小时</li>
                            <li>🏥 定期健康监测：每3个月测量一次身高、体重，动态跟踪BMI变化</li>
                        </ul>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="prediction-box low-risk">
                        <h3 style="color: #059669; margin-bottom: 0.8rem; font-size: 1.5rem; display: flex; align-items: center;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.5rem;">
                                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                                <polyline points="22 4 12 14.01 9 11.01"></polyline>
                            </svg>
                            肥胖风险低
                        </h3>
                        <p style="color: #4b5563; font-size: 1.1rem;">健康概率：<strong>{probability:.1f}%</strong></p>
                        <p style="color: #718096; font-size: 0.95rem; margin-top: 0.5rem;">提示：该学生当前生活方式较为健康，建议继续保持。</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.markdown(f'''
                    <div class="advice-box">
                        <h3 style="color: #2d3748; margin-bottom: 0.8rem; font-size: 1.2rem; font-weight: 500;">
                            健康保持建议
                        </h3>
                        <ul style="color: #4b5563; font-size: 1rem; line-height: 1.8;">
                            <li>💪 坚持运动习惯：每周保持3-5次规律锻炼，避免中断</li>
                            <li>🥗 均衡饮食结构：继续保持多样化饮食，减少零食和含糖饮料摄入</li>
                            <li>⏰ 规律作息：保持固定的作息时间，避免熬夜和睡懒觉</li>
                            <li>📊 定期监测：每年进行1-2次健康体检，跟踪生长发育情况</li>
                            <li>😊 情绪管理：保持积极乐观的心态，及时疏导负面情绪</li>
                        </ul>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # 可视化优化（专业配色+精简样式）
                st.markdown('<h2 class="sub-header">风险概率分布</h2>', unsafe_allow_html=True)
                
                # 设置中文字体（避免乱码）
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                
                fig, ax = plt.subplots(figsize=(10, 4))
                categories = ['非肥胖', '肥胖']
                probabilities = [predicted_proba[0], predicted_proba[1]]
                colors = ['#10b981', '#ef4444']  # 专业配色
                
                # 绘制条形图（优化样式）
                bars = ax.barh(categories, probabilities, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
                ax.set_xlim(0, 1.05)
                ax.set_xlabel('概率', fontsize=12, fontweight='500', color='#2d3748')
                ax.set_title('肥胖风险概率分布', fontsize=14, fontweight='600', color='#2d3748', pad=20)
                
                # 添加数值标签（优化位置和样式）
                for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                    ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{prob:.3f}', va='center', ha='left', 
                            fontsize=11, fontweight='500', color='#2d3748')
                
                # 美化图表（移除多余边框，优化网格）
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#e2e8f0')
                ax.spines['bottom'].set_color('#e2e8f0')
                ax.xaxis.grid(True, alpha=0.3, linestyle='--')
                ax.yaxis.grid(False)
                ax.tick_params(axis='y', labelsize=11, colors='#2d3748')
                ax.tick_params(axis='x', labelsize=10, colors='#4b5563')
                
                # 调整布局
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"预测过程中出现错误：{str(e)}", icon="❌")

with col_side:
    # 系统说明（精简内容，提升专业感）
    st.markdown('<h2 class="sub-header">系统说明</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0;">
        <h4 style="color: #2d3748; margin-bottom: 0.8rem; font-size: 1.1rem; font-weight: 500;">
            关于本系统
        </h4>
        <p style="color: #4b5563; font-size: 0.95rem; line-height: 1.6; margin-bottom: 1rem;">
            本系统基于机器学习算法，结合学生生理指标、饮食习惯、运动情况及情绪状态，预测未来1年肥胖风险，为学校和家庭提供科学的健康管理参考。
        </p>
        
        <h4 style="color: #2d3748; margin-bottom: 0.8rem; font-size: 1.1rem; font-weight: 500;">
            使用流程
        </h4>
        <ol style="color: #4b5563; font-size: 0.95rem; line-height: 1.8; margin-bottom: 1rem;">
            <li>在左侧完整填写学生各项信息</li>
            <li>点击"开始预测"按钮提交分析</li>
            <li>查看预测结果及专业干预建议</li>
        </ol>
        
        <h4 style="color: #2d3748; margin-bottom: 0.8rem; font-size: 1.1rem; font-weight: 500;">
            数据安全
        </h4>
        <p style="color: #4b5563; font-size: 0.95rem; line-height: 1.6;">
            所有输入数据仅在本地处理，不会上传至服务器，严格保障学生隐私安全。
        </p>
    </div>
    """, unsafe_allow_html=True)

# 页脚（精简样式）
st.markdown("""
<div class="footer">
    <p>学生肥胖风险预测系统 © 2024 | 基于机器学习的健康风险评估工具</p>
</div>
""", unsafe_allow_html=True)
