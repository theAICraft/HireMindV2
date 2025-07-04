import streamlit as st
from HireMind import HireMind

config = {
    'page_title': "HireMind",
    'page_icon': '🧠',
    'page_layout': 'wide',
    'groq_model': 'llama-3.3-70b-versatile',
    'openai_model': 'gpt-4o-mini',
    'radio_opt': ['GROQ', 'OpenAI']
}

# Page Setup
st.set_page_config(page_title=config['page_title'], page_icon=config['page_icon'], layout=config['page_layout'])

# Sidebar Controls
selected_opt = st.sidebar.radio("Choose the LLM provider", config['radio_opt'])
api_key = st.sidebar.text_input(f"Enter your {selected_opt} API Key", type="password")
model_name = config['openai_model'] if selected_opt == "OpenAI" else config['groq_model']
st.sidebar.text_input("Model", value=model_name, disabled=True)

# Cache model loading to avoid reprocessing
@st.cache_resource(show_spinner="⏳ Loading model...")
def load_and_test_model(api_key, model_name, is_openai):
    core = HireMind()
    core.load_model(model_name=model_name, api_key=api_key, openai=is_openai)
    core.test_model()
    return core

if api_key:
    try:
        core = load_and_test_model(api_key, model_name, selected_opt == "OpenAI")
        uploaded_resume = st.sidebar.file_uploader("Upload Resume (PDF, DOCX)", type=["pdf", "docx"])
        jd_text = st.sidebar.text_area("Paste the Job Description here")

        if uploaded_resume and jd_text.strip():
            st.markdown("## 📊 Resume Analysis")
            col1, col2 = st.columns(2)

            with st.spinner("🧠 Processing resume..."):
                resume_text = core.read_resume(uploaded_resume)
                analysis = core.analyse_resume(resume_text, jd_text)

            with col1:
                st.markdown("### 📄 Extracted Resume Content")
                st.subheader("🧑‍💼 Summary")
                st.write(resume_text.get('summary', 'Not provided'))

                st.subheader("📊 Experience")
                st.write(f"{resume_text.get('experience', 0)} Years")

                st.subheader("🛠️ Skills")
                for category, items in resume_text.get('skills', {}).items():
                    st.markdown(f"**{category.replace('_', ' ').title()}**: {', '.join(items)}")

            with col2:
                st.markdown("### 🎯 AI Analysis vs JD")
                st.subheader("💬 Review")
                st.write(analysis.get('review', 'No review provided'))
                st.subheader("📈 Fit Score (out of 10)")
                st.markdown(
                    f"""
                    <div style="width: 160px; height: 160px; border-radius: 80px; background: conic-gradient(#4caf50 {analysis['candidate_fit_score']*36}deg, #ddd {analysis['candidate_fit_score']*36}deg); display: flex; align-items: center; justify-content: center; font-size: 24px; font-weight: bold;">
                    {analysis['candidate_fit_score']}
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
                st.subheader("✅ Final Decision")
                decision = analysis.get('final_decision', '')
                if decision == 'R':
                    st.markdown(
                        "<div style='background-color:#f44336;color:white;padding:10px 20px;border-radius:10px;display:inline-block;font-weight:bold;'>Rejected</div>",
                        unsafe_allow_html=True)
                elif decision == 'A':
                    st.markdown(
                        "<div style='background-color:#4caf50;color:white;padding:10px 20px;border-radius:10px;display:inline-block;font-weight:bold;'>Accepted</div>",
                        unsafe_allow_html=True
                        )

    except Exception as e:
        with st.expander("❌ Model loading failed. Click to view error details."):
            st.error(str(e))
