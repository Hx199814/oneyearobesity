import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go  # 引入 Plotly 用于高级绘图

# 1. 页面配置：设置更专业的标题和布局
st.set_page_config(
    page_title="学生健康风险评估系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 自定义 CSS：提升界面质感（卡片化、字体优化、隐藏默认菜单）
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
        border-left: 5px solid #007bff;
    }
    
    /* 调整 metric 样式 */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* 按钮样式优化 */
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
        # 为了演示效果，如果没有模型文件，这里不会报错，而是返回 None
        # 实际使用请确保文件存在
        return None

model = load_model()

# --- 选项定义 (保持原有逻辑，去除文字中的 Emoji) ---
GENDER_options = {1: '男生', 2: '女生'}
D2_options = {1: '没有或偶尔', 2: '有时', 3: '时常或一半时间', 4: '多数时间或持续', 5: '不清楚'}
# ... (其他选项保持不变，省略重复代码以节省篇幅，逻辑与原代码一致) ...
# 这里为了代码完整性，我会把用到的选项补全
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
    
    # 简化逻辑展示，保持原有的判定标准
    thresholds = {
        6: (17.7, 17.5), 6.5: (18.1, 18.0), 7: (18.7, 18.5), 7.5: (19.2, 19.0),
        8: (19.7, 19.4), 8.5: (20.3, 19.9), 9: (20.8, 20.4), 9.5: (21.4, 21.0),
        10: (21.9, 21.5), 10.5: (22.5, 22.1), 11: (23.0, 22.7), 11.5: (23.6, 23.3),
        12: (24.1, 23.9), 12.5: (24.7, 24.5), 13: (25.2, 25.6), 13.5: (25.7, 25.6),
        14: (26.1, 25.9), 14.5: (26.4, 26.3), 15: (26.6, 26.6), 15.5: (26.9, 26.9),
        16: (27.1, 27.1), 16.5: (27.4, 27.4), 17: (27.6, 27.6), 17.5: (27.8, 27.8),
        18: (28.0, 28.0)
    }
    
    lookup_age = int(age * 2) / 2  # Round to nearest 0.5
    if lookup_age >= 18: lookup_age = 18
    if lookup_age < 6: lookup_age = 6
    
    if lookup_age in thresholds:
        limit = thresholds[lookup_age][0] if gender_code == 1 else thresholds[lookup_age][1]
        return 1 if bmi >= limit else 0
    return 0 if bmi < 28.0 else 1

# --- 侧边栏设计 ---
with st.sidebar:
    st.markdown("## 📋 评估参数录入")
    st.markdown("请在下方完善学生的相关信息。")
    st.divider()

    # 使用 Expander 折叠分类，界面更清爽
    with st.expander("基础生理指标", expanded=True):
        col1, col2 = st.columns(2)
        GENDER = col1.selectbox("性别", options=list(GENDER_options.keys()), format_func=lambda x: GENDER_options[x])
        AGE = col2.selectbox("年龄", options=list(range(6, 19)), format_func=lambda x: f"{x}岁")
        
        col3, col4 = st.columns(2)
        height_cm = col3.number_input("身高 (cm)", 100.0, 200.0, 150.0, 0.1)
        weight_kg = col4.number_input("体重 (kg)", 20.0, 100.0, 45.0, 0.1)
        
        # 实时 BMI 显示 (更隐蔽专业)
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

    with st.expander("生活方式与饮食"):
        PEC = st.selectbox("每周体育课节数", options=list(PEC_options.keys()), format_func=lambda x: PEC_options[x])
        FrFF = st.selectbox("水果摄入频率 (7天内)", options=list(FrFF_options.keys()), format_func=lambda x: FrFF_options[x])
        DVT = st.selectbox("蔬菜摄入种类 (每天)", options=list(DVT_options.keys()), format_func=lambda x: DVT_options[x])

    with st.expander("心理健康状况"):
        D1 = st.selectbox("受过往琐事困扰", options=list(D1_options.keys()), format_func=lambda x: D1_options[x])
        D2 = st.selectbox("食欲不振/胃口差", options=list(D2_options.keys()), format_func=lambda x: D2_options[x])
        D3 = st.selectbox("感到情绪低落/苦闷", options=list(D3_options.keys()), format_func=lambda x: D3_options[x])
        D9 = st.selectbox("感到生活无价值", options=list(D9_options.keys()), format_func=lambda x: D9_options[x])
        D11 = st.selectbox("睡眠质量差/不解乏", options=list(D11_options.keys()), format_func=lambda x: D11_options[x])
        D17 = st.selectbox("情绪失控/痛哭", options=list(D17_options.keys()), format_func=lambda x: D17_options[x])

    with st.expander("行为与其他"):
        HU = st.selectbox("长时间使用耳机 (>30分)", options=list(HU_options.keys()), format_func=lambda x: HU_options[x])
        FF = st.selectbox("过去12个月有打架行为", options=list(FF_options.keys()), format_func=lambda x: FF_options[x])
        PPP = st.selectbox("近期受到严厉责罚", options=list(PPP_options.keys()), format_func=lambda x: PPP_options[x])

# --- 主页面区域 ---

# 头部设计
st.markdown("### 学生健康风险智能评估系统")
st.markdown("基于 CatBoost 机器学习模型的多维度健康数据分析平台")
st.divider()

# 预测逻辑
if st.button("开始智能分析", type="primary", use_container_width=True):
    if model is None:
        st.error("系统提示：模型文件未加载，请联系管理员检查服务器配置。")
    else:
        with st.spinner("正在进行多维数据计算..."):
            try:
                # 构建特征向量
                feature_values = [GENDER, baseline_obesity, D2, AGE, D1, D9, HU, D11, PEC, FrFF, D17, DVT, FF, D3, PPP]
                features = np.array([feature_values], dtype=np.float32)
                
                # 预测
                predicted_class = int(model.predict(features)[0])
                predicted_proba = model.predict_proba(features)[0]
                
                # 获取目标概率
                if 0 <= predicted_class < len(predicted_proba):
                    probability = predicted_proba[predicted_class] * 100
                    risk_score = predicted_proba[1] * 100 # 专门获取“肥胖/高风险”的概率用于仪表盘
                else:
                    probability = 0
                    risk_score = 0
                
                # --- 结果展示区 ---
                
                col_metrics, col_viz = st.columns([1.2, 1])
                
                with col_metrics:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    if predicted_class == 1:
                        st.markdown("#### 🔴 评估结果：高风险")
                        st.markdown("根据模型分析，该学生存在较高的肥胖或相关健康风险。")
                        st.divider()
                        st.metric("风险指数", f"{probability:.1f}%", delta="+高危", delta_color="inverse")
                    else:
                        st.markdown("#### 🟢 评估结果：低风险")
                        st.markdown("各项指标处于相对健康范围，请继续保持良好的生活习惯。")
                        st.divider()
                        st.metric("健康指数", f"{probability:.1f}%", delta="稳定")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # 建议部分使用原生 Expanders 或纯文本，保持简洁
                    st.subheader("💡 综合干预建议")
                    if predicted_class == 1:
                        st.info("建议重点关注饮食结构调整与运动量提升。")
                        st.markdown("""
                        * **运动处方**：每日中高强度运动（MVPA）累计至少 60 分钟。
                        * **营养干预**：严格限制含糖饮料，增加膳食纤维摄入。
                        * **睡眠管理**：保障 8-10 小时优质睡眠，建立规律生物钟。
                        """)
                    else:
                        st.success("建议维持当前健康的生活方式。")
                        st.markdown("""
                        * **持续监测**：每季度进行一次生长发育指标监测。
                        * **习惯维持**：保持“三餐规律、定期运动”的优良习惯。
                        """)

                with col_viz:
                    # 使用 Plotly 绘制专业仪表盘
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_score,
                        title = {'text': "肥胖风险评估值", 'font': {'size': 18, 'color': "#555"}},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        number = {'suffix': "%", 'font': {'size': 26}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#999"},
                            'bar': {'color': "#e63946" if risk_score > 50 else "#2a9d8f"}, # 动态颜色
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

            except Exception as e:
                st.error(f"分析过程中发生系统错误: {str(e)}")
else:
    # 空状态页展示，引导用户操作
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #6c757d;">
        <h4>👈 请在左侧面板输入完整信息</h4>
        <p>点击上方按钮启动 AI 预测模型</p>
    </div>
    """, unsafe_allow_html=True)

# 页脚设计
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #adb5bd; font-size: 0.8rem;'>
    Student Health Risk Assessment System © 2025 | Powered by CatBoost & Streamlit
</div>
""", unsafe_allow_html=True)
