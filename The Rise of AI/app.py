import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="AI & Automation Dashboard", layout="wide", initial_sidebar_state="expanded")

a = pd.read_csv("The Rise Of Artificial Intellegence2.csv")  

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {background-color: #2B2B2B; color: white;}
    h1, h2, h3 {color: #FF4B4B;}
    .stButton > button {background-color: #FF6347; color: white; border-radius: 10px;}
    .stButton > button:hover {background-color: #FF7F50;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 AI and Automation Dataset Analysis + ML Prediction")

st.sidebar.header("📂 Navigation")
section = st.sidebar.radio(
    "Choose Section",
    [
        "Overview", "Trend Analysis", "Job Impact", "AI Adoption",
        "AI in Healthcare", "Risk of Automation", "Revenue & Productivity",
        "Voice Assistants", "AI in Strategy", "Word Cloud",
        "Column-wise Analysis", "AI Revenue Prediction (ML)"
    ],
)

a_cleaned = a.copy()
percentage_columns = [
    'AI Adoption (%)', 'Global Expectation for AI Adoption (%)',
    'Estimated Jobs Eliminated by AI (millions)', 'Estimated New Jobs Created by AI (millions)',
    'Net Job Loss in the US', 'Organizations Believing AI Provides Competitive Edge',
    'Companies Prioritizing AI in Strategy', 'Marketers Believing AI Improves Email Revenue',
    'Expected Increase in Employee Productivity Due to AI (%)', 'Americans Using Voice Assistants (%)',
    'Medical Professionals Using AI for Diagnosis',
    'Jobs at High Risk of Automation - Transportation & Storage (%)',
    'Jobs at High Risk of Automation - Wholesale & Retail Trade',
    'Jobs at High Risk of Automation - Manufacturing'
]

for col in percentage_columns:
    a_cleaned[col] = a_cleaned[col].str.replace('%', '').astype(float)

if section == "AI Revenue Prediction (ML)":
    st.subheader("📊 AI Software Revenue Prediction Using ML Models")

    ba = a_cleaned.select_dtypes(include=[np.number])
    x = ba.drop(columns=['AI Software Revenue(in Billions)'])
    y = ba['AI Software Revenue(in Billions)']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    algo = st.sidebar.selectbox("⚙️ Select Algorithm", ["Linear Regression", "Random Forest"])

    if algo == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.metric(label="Algorithm", value=algo)
    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
    st.metric(label="R² Score (Accuracy)", value=f"{r2:.2f}")

    results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    fig = px.scatter(results_df, x="Actual", y="Predicted", 
                     title=f"📈 Actual vs Predicted AI Software Revenue ({algo})",
                     template="plotly_dark", trendline="ols")
    fig.add_shape(
        type="line", line=dict(dash="dash", color="red"), 
        x0=y_test.min(), y0=y_test.min(), 
        x1=y_test.max(), y1=y_test.max()
    )
    st.plotly_chart(fig, use_container_width=True)

elif section == "Word Cloud":
    st.subheader("🌐 Word Cloud of AI-related Jobs")
    counts = Counter(a_cleaned["Jobs at High Risk of Automation - Manufacturing"].dropna().astype(str))
    wordcloud = WordCloud(stopwords=set(STOPWORDS), background_color="black").generate_from_frequencies(counts)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

elif section == "Column-wise Analysis":
    selected_column = st.selectbox("📋 Select Column", a.columns)
    if a[selected_column].dtype in ['int64', 'float64']:
        fig = px.histogram(a, x=selected_column, title=f"Distribution of {selected_column}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        column_counts = a[selected_column].value_counts().reset_index()
        column_counts.columns = [selected_column, 'count']
        fig = px.bar(column_counts, x=selected_column, y='count', title=f"Distribution of {selected_column}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)


elif section == "Trend Analysis":
    st.subheader("📊 Trend Analysis of AI Software Revenue and Global Market Value")
    fig = px.line(a_cleaned, x='Year', y=['AI Software Revenue(in Billions)', 'Global AI Market Value(in Billions)'],
                  markers=True, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
elif section == "Job Impact":
    st.subheader("👨‍💻 Job Impact Analysis: Jobs Eliminated vs. Created")
    fig = px.line(a_cleaned, x='Year', 
                  y=['Estimated Jobs Eliminated by AI (millions)', 'Estimated New Jobs Created by AI (millions)'],
                  markers=True, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif section == "AI in Healthcare":
    st.subheader("🏥 AI in Healthcare: Diagnosis and Contribution")
    fig = px.line(a_cleaned, x='Year', 
                  y=['Medical Professionals Using AI for Diagnosis', 'AI Contribution to Healthcare(in Billions)'],
                  markers=True, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif section == "Risk of Automation":
    st.subheader("⚠️ Jobs at High Risk of Automation")
    fig = px.line(a_cleaned, x='Year', 
                  y=[
                      'Jobs at High Risk of Automation - Transportation & Storage (%)',
                      'Jobs at High Risk of Automation - Wholesale & Retail Trade',
                      'Jobs at High Risk of Automation - Manufacturing'
                  ], markers=True, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
