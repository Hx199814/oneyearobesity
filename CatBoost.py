import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 【优化1】设置Matplotlib后端为非交互式，提升Web端绘图速度并防止内存泄漏
matplotlib.use('Agg')

# 页面配置
st.set_page_config(
    page_title="学生肥胖风险预测系统",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载模型 (保持缓存)
@st.cache_resource
def load_model():
    try:
        return joblib.load('CatBoost.pkl')
    except FileNotFoundError:
        st.error("错误：未找到模型文件 'CatBoost.pkl'，请确保文件在当前目录下。")
        return None

model = load_model()

# --- 保持原有变量定义不变 ---
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

# 计算基线肥胖函数 (保持逻辑不变)
def calculate_baseline_obesity(age, gender, height_cm, weight_kg):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    gender_code = 1 if gender == 1 else 0
    
    # 逻辑判断保持原样，仅做结构折叠以节省视觉空间
    if age >= 6 and age < 6.5: return 1 if (gender_code == 1 and bmi >= 17.7) or (gender_code == 0 and bmi >= 17.5) else 0
    elif age >= 6.5 and age < 7: return 1 if (gender_code == 1 and bmi >= 18.1) or (gender_code == 0 and bmi >= 18.0) else 0
    elif age >= 7 and age < 7.5: return 1 if (gender_code == 1 and bmi >= 18.7) or (gender_code == 0 and bmi >= 18.5) else 0
    elif age >= 7.5 and age < 8: return 1 if (gender_code == 1 and bmi >= 19.2) or (gender_code == 0 and bmi >= 19.0) else 0
    elif age >= 8 and age < 8.5: return 1 if (gender_code == 1 and bmi >= 19.7) or (gender_code == 0 and bmi >= 19.4) else 0
    elif age >= 8.5 and age < 9: return 1 if (gender_code == 1 and bmi >= 20.3) or (gender_code == 0 and bmi >= 19.9) else 0
    elif age >= 9 and age < 9.5: return 1 if (gender_code == 1 and bmi >= 20.8) or (gender_code == 0 and bmi >= 20.4) else 0
    elif age >= 9.5 and age < 10: return 1 if (gender_code == 1 and bmi >= 21.4) or (gender_code == 0 and bmi >= 21.0) else 0
    elif age >= 10 and age < 10.5: return 1 if (gender_code == 1 and bmi >= 21.9) or (gender_code == 0 and bmi >= 21.5) else 0
    elif age >= 10.5 and age < 11: return 1 if (gender_code == 1 and bmi >= 22.5) or (gender_code == 0 and bmi >= 22.1) else 0
    elif age >= 11 and age < 11.5: return 1 if (gender_code == 1 and bmi >= 23.0) or (gender_code == 0 and bmi >= 22.7) else 0
    elif age >= 11.5 and age < 12: return 1 if (gender_code == 1 and bmi >= 23.6) or (gender_code == 0 and bmi >= 23.3) else 0
    elif age >= 12 and age < 12.5: return 1 if (gender_code == 1 and bmi >= 24.1) or (gender_code == 0 and bmi >= 23.9) else 0
    elif age >= 12.5 and age < 13: return 1 if (gender_code == 1 and bmi >= 24.7) or (gender_code == 0 and bmi >= 24.5) else 0
    elif age >= 13 and age < 13.5: return 1 if (gender_code == 1 and bmi >= 25.2) or (gender_code == 0 and bmi >= 25.6) else 0
    elif age >= 13.5 and age < 14: return 1 if (gender_code == 1 and bmi >= 25.7) or (gender_code == 0 and bmi >= 25.6) else 0
    elif age >= 14 and age < 14.5: return 1 if (gender_code == 1 and bmi >= 26.1) or (gender_code == 0 and bmi >= 25.9) else 0
    elif age >= 14.5 and age < 15: return 1 if (gender_code == 1 and bmi >= 26.4) or (gender_code == 0 and bmi >= 26.3) else 0
    elif age >= 15 and age < 15.5: return 1 if (gender_code == 1 and bmi >= 26.6) or (gender_code == 0 and bmi >= 26.6) else 0
    elif age >= 15.5 and age < 16: return 1 if (gender_code == 1 and bmi >= 26.9) or (gender_code == 0 and bmi >= 26.9) else 0
    elif age >= 16 and age < 16.5: return 1 if (gender_code == 1 and bmi >= 27.1) or (gender_code == 0 and bmi >= 27.1) else 0
    elif age >= 16.5 and age < 17: return 1 if (gender_code == 1 and bmi >= 27.4) or (gender_code == 0 and bmi >= 27.4) else 0
    elif age >= 17 and age < 17.5: return 1 if (gender_code == 1 and bmi >= 27.6) or (gender_code == 0 and bmi >= 27.6) else 0
    elif age >= 17.5 and age < 18: return 1 if (gender_code == 1 and bmi >= 27.8) or (gender_code == 0 and bmi >= 27.8) else 0
    elif age >= 18: return 1 if bmi >= 28.0 else 0
    return 0

# --- 侧边栏：输入区域 (保持不变) ---
with st.sidebar:
    st.header("📝 学生信息录入")
    
    st.markdown("### 👤 基本信息")
    col1, col2 = st.columns(2)
    with col1:
        GENDER = st.selectbox("性别:", options=list(GENDER_options.keys()), format_func=lambda x: GENDER_options[x])
    with col2:
        AGE = st.selectbox("年龄:", options=list(range(6, 19)), format_func=lambda x: f"{x}岁")
    
    st.markdown("### 📊 身高体重")
    col1, col2 = st.columns(2)
    with col1:
        height_cm = st.number_input("身高 (cm):", 100.0, 200.0, 150.0, 0.1)
    with col2:
        weight_kg = st.number_input("体重 (kg):", 20.0, 100.0, 45.0, 0.1)
    
    # 实时计算基线 (计算极快，不影响性能)
    baseline_obesity = calculate_baseline_obesity(AGE, GENDER, height_cm, weight_kg)
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    # 使用原生组件展示状态
    st.info(f"当前 BMI: {bmi:.1f} | 基线状态: {'肥胖' if baseline_obesity == 1 else '正常'}")
    
    st.markdown("### 🍎 饮食与运动")
    PEC = st.selectbox("每周体育课节数:", options=list(PEC_options.keys()), format_func=lambda x: PEC_options[x])
    FrFF = st.selectbox("过去七天吃水果次数:", options=list(FrFF_options.keys()), format_func=lambda x: FrFF_options[x])
    DVT = st.selectbox("每天吃几种蔬菜:", options=list(DVT_options.keys()), format_func=lambda x: DVT_options[x])
    
    st.markdown("### 😊 情绪状态")
    D1 = st.selectbox("以前不困扰的事现在烦恼:", options=list(D1_options.keys()), format_func=lambda x: D1_options[x])
    D2 = st.selectbox("胃口不好:", options=list(D2_options.keys()), format_func=lambda x: D2_options[x])
    D3 = st.selectbox("无法摆脱苦闷:", options=list(D3_options.keys()), format_func=lambda x: D3_options[x])
    D9 = st.selectbox("生活一无是处:", options=list(D9_options.keys()), format_func=lambda x: D9_options[x])
    D11 = st.selectbox("睡后不解乏:", options=list(D11_options.keys()), format_func=lambda x: D11_options[x])
    D17 = st.selectbox("曾放声痛哭:", options=list(D17_options.keys()), format_func=lambda x: D17_options[x])
    
    st.markdown("### 📱 行为习惯")
    HU = st.selectbox("使用耳机(>30分):", options=list(HU_options.keys()), format_func=lambda x: HU_options[x])
    FF = st.selectbox("过去12个月打架:", options=list(FF_options.keys()), format_func=lambda x: FF_options[x])
    PPP = st.selectbox("过去30天被家长打骂:", options=list(PPP_options.keys()), format_func=lambda x: PPP_options[x])

# --- 主页面区域 (优化后) ---
st.title("🏫 学生肥胖风险预测系统")
st.markdown("---")

# 预测按钮逻辑
if st.button("🚀 开始预测", type="primary", use_container_width=True):
    if model is None:
        st.error("模型未加载，无法预测。")
    else:
        with st.spinner("正在分析数据..."):
            try:
                # 准备数据
                feature_values = [GENDER, baseline_obesity, D2, AGE, D1, D9, HU, D11, PEC, FrFF, D17, DVT, FF, D3, PPP]
                features = np.array([feature_values])
                
                # 预测
                predicted_class = model.predict(features)[0] 
                predicted_proba = model.predict_proba(features)[0]
                probability = predicted_proba[predicted_class] * 100
                
                # 【优化2】使用原生容器布局，更轻量
                col_result, col_chart = st.columns([1, 1])
                
                with col_result:
                    st.subheader("📋 分析结果")
                    if predicted_class == 1:
                        st.error(f"⚠️ **风险提示：高风险**")
                        st.metric("肥胖风险概率", f"{probability:.1f}%", delta="注意", delta_color="inverse")
                        st.markdown("**建议：** 增加每日运动量至60分钟，严格控制糖分摄入，并保证充足睡眠。")
                    else:
                        st.success(f"✅ **风险提示：低风险**")
                        st.metric("健康维持概率", f"{probability:.1f}%", delta="保持")
                        st.markdown("**建议：** 继续保持目前的饮食和运动习惯，定期监测身高体重。")

                with col_chart:
                    # 【优化3】简化的绘图逻辑，避免复杂的CSS注入
                    st.subheader("📈 概率分布")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    categories = ['健康', '肥胖风险']
                    probs = [predicted_proba[0], predicted_proba[1]]
                    colors = ['#28a745', '#dc3545']
                    
                    ax.barh(categories, probs, color=colors, alpha=0.8, height=0.5)
                    ax.set_xlim(0, 1)
                    # 隐藏边框，让图表更干净
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.set_xticks([]) # 隐藏X轴刻度
                    
                    # 标注数值
                    for i, v in enumerate(probs):
                        ax.text(v + 0.02, i, f'{v*100:.1f}%', va='center', fontweight='bold')
                    
                    st.pyplot(fig)
                    plt.close(fig) # 【重要】显式关闭图表释放内存

            except Exception as e:
                st.error(f"预测出错: {str(e)}")
else:
    # 默认状态显示
    st.info("👈 请在左侧侧边栏填写完整信息，然后点击上方按钮开始预测。")

# 简单的页脚
st.markdown("---")
st.caption("学生肥胖风险预测系统 © 2025 | 数据仅供参考")
