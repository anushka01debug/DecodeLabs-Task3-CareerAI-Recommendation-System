# -*- coding: utf-8 -*-
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import re
import os
import pypdf
import json
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareerAI — Find Your Path",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# API KEY & CONFIGURATION
# ─────────────────────────────────────────────────────────
API_KEY = ""  # Hardcoded default Gemini key

# ─────────────────────────────────────────────────────────
# THEME CONFIGURATION
# ─────────────────────────────────────────────────────────
theme_choice = st.sidebar.selectbox(
    "🎨 App Visual Theme", 
    ["Space Dark (Violet)", "Orchid Light (Clean)"],
    help="Toggle between premium dark mode and high-contrast light mode styling."
)

if theme_choice == "Space Dark (Violet)":
    main_bg = "#090d16"
    sidebar_bg = "#0d1424"
    card_bg = "rgba(255, 255, 255, 0.03)"
    card_border = "rgba(255, 255, 255, 0.08)"
    card_border_hover = "#a855f7"
    text_color = "#f3f4f6"
    sub_text_color = "#9ca3af"
    accent_color = "#a855f7"
    accent_gradient = "linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #f43f5e 100%)"
    plotly_template = "plotly_dark"
    grid_color = "rgba(255, 255, 255, 0.05)"
    badge_matched_bg = "rgba(104, 211, 145, 0.08)"
    badge_matched_border = "rgba(104, 211, 145, 0.25)"
    badge_matched_text = "#68d391"
    badge_missing_bg = "rgba(252, 129, 74, 0.08)"
    badge_missing_border = "rgba(252, 129, 74, 0.25)"
    badge_missing_text = "#fc814a"
    stat_box_bg = "rgba(255, 255, 255, 0.02)"
    stat_box_hover = "rgba(168, 85, 247, 0.1)"
    primary_btn_text = "#090d16"
    bar_fill_color = "linear-gradient(to right, #a855f7, #ec4899)"
else:
    main_bg = "#f3f4f6"
    sidebar_bg = "#ffffff"
    card_bg = "rgba(255, 255, 255, 0.85)"
    card_border = "rgba(124, 58, 237, 0.12)"
    card_border_hover = "#7c3aed"
    text_color = "#111827"
    sub_text_color = "#4b5563"
    accent_color = "#7c3aed"
    accent_gradient = "linear-gradient(135deg, #7c3aed 0%, #db2777 100%)"
    plotly_template = "plotly"
    grid_color = "rgba(0, 0, 0, 0.05)"
    badge_matched_bg = "rgba(16, 185, 129, 0.1)"
    badge_matched_border = "rgba(16, 185, 129, 0.3)"
    badge_matched_text = "#047857"
    badge_missing_bg = "rgba(245, 158, 11, 0.1)"
    badge_missing_border = "rgba(245, 158, 11, 0.3)"
    badge_missing_text = "#b45309"
    stat_box_bg = "rgba(0, 0, 0, 0.02)"
    stat_box_hover = "rgba(124, 58, 237, 0.08)"
    primary_btn_text = "#ffffff"
    bar_fill_color = "linear-gradient(to right, #7c3aed, #db2777)"

# Inject global premium styles
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}}

.stApp {{
    background-color: {main_bg} !important;
    color: {text_color} !important;
    min-height: 100vh;
}}

.main, .block-container {{
    position: relative;
    z-index: 2;
}}

header, footer, #MainMenu {{
    visibility: hidden;
}}

.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 5rem !important;
    max-width: 1000px !important;
    word-wrap: break-word !important;
}}

/* Premium Glassmorphic Card styling */
.premium-card {{
    background-color: {card_bg};
    border-radius: 16px;
    padding: 24px;
    border: 1px solid {card_border};
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    color: {text_color};
}}

.premium-card:hover {{
    border-color: {card_border_hover};
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07);
    transform: translateY(-2px);
}}

/* Style bordered containers as premium cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {card_bg} !important;
    border-radius: 16px !important;
    border: 1px solid {card_border} !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    padding: 24px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    color: {text_color} !important;
}}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    border-color: {card_border_hover} !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07) !important;
    transform: translateY(-2px) !important;
}}

/* Ensure child block containers do not add default borders */
div[data-testid="stVerticalBlockBorderWrapper"] > div {{
    border: none !important;
}}

/* Metric numbers */
.metric-value {{
    font-size: 2.2rem;
    font-weight: 700;
    color: {accent_color};
}}

.metric-label {{
    font-size: 0.9rem;
    color: {sub_text_color};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}}

/* Sidebar overrides */
section[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
    border-right: 1px solid {card_border} !important;
}}

/* Custom text input and selectbox styling */
div[data-testid="stTextInput"] input {{
    background: {sidebar_bg} !important;
    color: {text_color} !important;
    border: 1.5px solid {card_border} !important;
    border-radius: 14px !important;
    height: 52px !important;
    font-size: 15px !important;
    padding: 0 16px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.2s !important;
}}

div[data-testid="stTextInput"] input:focus {{
    border-color: {accent_color} !important;
    box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.15) !important;
    outline: none !important;
}}

div[data-testid="stSelectbox"]>div>div {{
    background: {sidebar_bg} !important;
    color: {text_color} !important;
    border: 1.5px solid {card_border} !important;
    border-radius: 12px !important;
    height: 48px !important;
}}

/* Main CTA button style */
div.stButton>button[kind="secondary"] {{
    width: 100% !important;
    height: 52px !important;
    border-radius: 14px !important;
    border: none !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    color: {primary_btn_text} !important;
    background: {accent_gradient} !important;
    box-shadow: 0 4px 20px rgba(168, 85, 247, 0.25) !important;
    transition: all 0.25s !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}}

div.stButton>button[kind="secondary"]:hover {{
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 30px rgba(236, 72, 153, 0.35) !important;
}}

/* Small Skill Tags inside columns */
div[data-testid="column"] .stButton>button {{
    width: 100% !important;
    height: 32px !important;
    border-radius: 999px !important;
    border: 1.2px solid {card_border} !important;
    background: rgba(168, 85, 247, 0.05) !important;
    color: {accent_color} !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    padding: 0 8px !important;
    transition: all 0.18s !important;
}}

div[data-testid="column"] .stButton>button:hover {{
    border-color: {accent_color} !important;
    background: rgba(168, 85, 247, 0.12) !important;
    color: {text_color} !important;
}}

/* Expander styling */
div[data-testid="stExpander"] {{
    background: {card_bg} !important;
    border: 1px solid {card_border} !important;
    border-radius: 16px !important;
    margin-bottom: 12px !important;
    backdrop-filter: blur(8px);
}}
div[data-testid="stExpander"] summary {{
    color: {accent_color} !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}}

/* Custom Skill badges */
.badge-have {{
    display: inline-block;
    background: {badge_matched_bg};
    border: 1.2px solid {badge_matched_border};
    color: {badge_matched_text};
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 12.5px;
    margin: 4px;
    font-weight: 600;
}}

.badge-need {{
    display: inline-block;
    background: {badge_missing_bg};
    border: 1.2px solid {badge_missing_border};
    color: {badge_missing_text};
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 12.5px;
    margin: 4px;
    font-weight: 600;
}}

/* Timeline layout */
.timeline-card {{
    display: flex;
    gap: 14px;
    margin-bottom: 14px;
    align-items: flex-start;
}}

.timeline-badge {{
    min-width: 58px;
    height: 58px;
    border-radius: 14px;
    background: {stat_box_bg};
    border: 1.5px solid {card_border};
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(4px);
}}

.timeline-wk {{
    font-size: 9px;
    color: {sub_text_color};
    letter-spacing: 0.8px;
    text-transform: uppercase;
    font-weight: 700;
}}

.timeline-num {{
    font-size: 17px;
    font-weight: 800;
    color: {accent_color};
}}

.timeline-content {{
    flex: 1;
    background: {stat_box_bg};
    border: 1px solid {card_border};
    border-radius: 14px;
    padding: 14px 18px;
    transition: all 0.2s;
}}

.timeline-content:hover {{
    border-color: {accent_color};
}}

.timeline-title {{
    font-size: 14.5px;
    font-weight: 700;
    color: {text_color};
    margin-bottom: 4px;
}}

.timeline-task {{
    font-size: 13px;
    color: {sub_text_color};
    line-height: 1.55;
    font-weight: 300;
}}

.priority-lbl {{
    display: inline-block;
    font-size: 9px;
    font-weight: 800;
    padding: 2px 7px;
    border-radius: 6px;
    margin-left: 8px;
    vertical-align: middle;
    text-transform: uppercase;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# RECOGNIZED TECHNICAL SKILLS REFERENCE
# ─────────────────────────────────────────────────────────
TECH_SKILLS = [
    "python", "machine learning", "deep learning", "pytorch", "tensorflow", "natural language processing", "nlp", 
    "computer vision", "transformers", "pandas", "numpy", "scikit-learn", "keras", "vector databases", "huggingface", 
    "generative ai", "llms", "r", "sql", "data visualization", "statistics", "tableau", "power bi", "matplotlib", 
    "seaborn", "data mining", "a/b testing", "predictive modeling", "javascript", "typescript", "react", "node.js", 
    "express", "html", "css", "mongodb", "postgresql", "git", "github", "docker", "restful apis", "next.js", "aws", 
    "tailwindcss", "linux", "bash", "kubernetes", "terraform", "ci/cd", "jenkins", "github actions", "ansible", 
    "prometheus", "grafana", "nginx", "cloud computing", "shell scripting", "azure", "gcp", "networking", "security", 
    "serverless", "microservices", "iam", "disaster recovery", "firewalls", "cryptography", "wireshark", 
    "penetration testing", "vulnerability assessment", "siem", "incident response", "security compliance", 
    "active directory", "cyber security", "angular", "vue.js", "web design", "responsive design", "webpack", "sass", 
    "redux", "java", "go", "c#", "fastapi", "django", "redis", "grpc", "system design", "spark", "hadoop", "kafka", 
    "etl", "data warehousing", "snowflake", "airflow", "scala", "data pipelines", "dbt", "bigquery", "swift", 
    "kotlin", "flutter", "react native", "dart", "objective-c", "ios development", "android development", 
    "mobile design", "solidity", "ethereum", "rust", "smart contracts", "web3", "hardhat", "distributed systems",
    "product strategy", "agile", "scrum", "roadmapping", "user research", "wireframing", "jira", "market analysis", 
    "communication", "product design", "figma", "adobe xd", "prototyping", "typography", "color theory", 
    "visual design", "user flows", "interaction design", "selenium", "qa automation", "cypress", "playwright", 
    "manual testing", "api testing", "postman", "c", "c++", "embedded systems", "microcontrollers", "rtos", 
    "assembly", "hardware design", "firmware", "i2c", "spi", "uart"
]

DOMAINS = {
    "AI & Data Science": [
        "python", "machine learning", "deep learning", "pytorch", "tensorflow", "nlp", "computer vision", 
        "transformers", "pandas", "numpy", "scikit-learn", "keras", "vector databases", "huggingface", 
        "generative ai", "llms", "r", "sql", "tableau", "power bi", "statistics", "data mining", "spark", 
        "hadoop", "kafka", "etl", "data warehousing", "snowflake", "airflow", "scala", "data pipelines", "bigquery"
    ],
    "Frontend Development": [
        "javascript", "typescript", "react", "html", "css", "tailwindcss", "next.js", "angular", "vue.js", 
        "web design", "responsive design", "webpack", "sass", "redux", "figma", "adobe xd", "wireframing", 
        "prototyping", "visual design", "user research", "typography", "color theory", "user flows", "interaction design"
    ],
    "Backend & Databases": [
        "node.js", "express", "mongodb", "postgresql", "sql", "redis", "java", "go", "c#", "fastapi", "django", 
        "grpc", "system design", "solidity", "ethereum", "rust", "smart contracts", "web3", "hardhat", "distributed systems"
    ],
    "Cloud & DevOps": [
        "docker", "kubernetes", "aws", "terraform", "ci/cd", "jenkins", "github actions", "ansible", 
        "prometheus", "grafana", "nginx", "cloud computing", "shell scripting", "azure", "gcp", "networking", 
        "serverless", "microservices", "iam", "disaster recovery"
    ],
    "System & Cybersecurity": [
        "linux", "bash", "firewalls", "cryptography", "wireshark", "penetration testing", "vulnerability assessment", 
        "siem", "incident response", "security compliance", "active directory", "cyber security", "embedded systems", 
        "microcontrollers", "rtos", "assembly", "hardware design", "firmware", "c", "c++", "i2c", "spi", "uart"
    ]
}

# ─────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────
for k, v in {
    "skills_input": "",
    "experience": "Mid-level (2–5 yrs)",
    "goal": "Switch career",
    "extracted_skills": [],
    "parsed_resume_text": "",
    "resume_name": "",
    "selected_career": None,
    "quiz_active": False,
    "quiz_step": 0,
    "quiz_answers": {},
    "chat_history": [],
    "ai_insights_cache": {},
    "last_profile_hash": "",
    "cat_filter": "All"
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def add_skill(skill):
    cur = st.session_state.skills_input.strip()
    existing = [s.strip().lower() for s in cur.split(",") if s.strip()]
    if skill.lower() not in existing:
        st.session_state.skills_input = (cur + ", " + skill).lstrip(", ")

# Automatic cache invalidation when key fields change
current_profile_hash = f"{st.session_state.skills_input}|{st.session_state.experience}|{st.session_state.goal}"
if st.session_state.last_profile_hash != current_profile_hash:
    st.session_state.ai_insights_cache = {}  # Invalidate cached AI roadmaps
    st.session_state.last_profile_hash = current_profile_hash

# ─────────────────────────────────────────────────────────
# ANIMATED BACKGROUND CANVAS (Dark Theme Only)
# ─────────────────────────────────────────────────────────
if theme_choice == "Space Dark (Violet)":
    components.html("""
    <!DOCTYPE html><html><head><style>
    html,body{margin:0;padding:0;overflow:hidden;background:transparent}
    .bg-wrap{position:fixed;inset:0;width:100vw;height:100vh;pointer-events:none;overflow:hidden}
    #sl{position:absolute;inset:0;z-index:0}
    .star{position:absolute;border-radius:50%;background:rgba(255,255,255,0.9);
        box-shadow:0 0 4px rgba(255,255,255,0.5),0 0 10px rgba(168,85,247,0.2);
        animation:tw ease-in-out infinite alternate;will-change:transform,opacity}
    @keyframes tw{0%{opacity:.12;transform:scale(.8)}50%{opacity:.85;transform:scale(1.3)}100%{opacity:.18;transform:scale(.9)}}
    #pc{position:absolute;inset:0;width:100vw;height:100vh;z-index:1}
    </style></head><body>
    <div class="bg-wrap"><div id="sl"></div><canvas id="pc"></canvas></div>
    <script>
    const sl=document.getElementById("sl"),cv=document.getElementById("pc"),ctx=cv.getContext("2d");
    let pts=[];const N=45;
    function rnd(a,b){return Math.random()*(b-a)+a}
    function resize(){cv.width=window.innerWidth;cv.height=window.innerHeight}
    function stars(){
        sl.innerHTML="";
        for(let i=0;i<75;i++){
            const s=document.createElement("div");s.className="star";
            const sz=rnd(.8,2.4);
            Object.assign(s.style,{width:sz+"px",height:sz+"px",left:rnd(0,window.innerWidth)+"px",
                top:rnd(0,window.innerHeight)+"px",animationDuration:rnd(1.8,5.2)+"s",
                animationDelay:rnd(0,4)+"s",opacity:rnd(.1,.55)});
            sl.appendChild(s);
        }
    }
    function mkpts(){pts=[];for(let i=0;i<N;i++)pts.push({x:Math.random()*cv.width,y:Math.random()*cv.height,vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.35,r:Math.random()*1.4+.6})}
    function draw(){
        ctx.clearRect(0,0,cv.width,cv.height);
        for(let i=0;i<pts.length;i++){
            const p=pts[i];p.x+=p.vx;p.y+=p.vy;
            if(p.x<0||p.x>cv.width)p.vx*=-1;if(p.y<0||p.y>cv.height)p.vy*=-1;
            ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
            ctx.fillStyle="rgba(168,85,247,.3)";ctx.fill();
            for(let j=i+1;j<pts.length;j++){
                const q=pts[j],dx=p.x-q.x,dy=p.y-q.y,d=Math.sqrt(dx*dx+dy*dy);
                if(d<115){ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);
                    ctx.strokeStyle=`rgba(168,85,247,${.065-d/1800})`;ctx.lineWidth=.65;ctx.stroke();}
            }
        }
        requestAnimationFrame(draw);
    }
    function init(){resize();stars();mkpts();draw()}
    window.addEventListener("resize",()=>{resize();stars();mkpts()});
    init();
    </script></body></html>
    """, height=0)

# ─────────────────────────────────────────────────────────
# DATA LOADING & SEEDING
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_career_data():
    if os.path.exists("skills.csv"):
        df = pd.read_csv("skills.csv")
    else:
        # Prepopulate comprehensive default dataset
        data = {
            "Role": [
                "Data Scientist", "Frontend Developer", "DevOps Engineer", "AI / Machine Learning Engineer",
                "Full Stack Developer", "Cloud Architect", "Cybersecurity Analyst", "Backend Engineer",
                "Data Engineer", "Mobile App Developer", "Blockchain Engineer", "Product Manager",
                "UI/UX Designer", "QA Automation Engineer", "Embedded Systems Engineer"
            ],
            "Skills": [
                "python statistics machine learning sql pandas numpy scikit-learn r data visualization statistics tableau matplotlib",
                "javascript react css html typescript tailwindcss next.js web design responsive design webpack angular",
                "docker kubernetes ci/cd linux bash aws terraform jenkins github actions ansible prometheus grafana nginx",
                "python machine learning deep learning pytorch tensorflow natural language processing nlp computer vision transformers vector databases generative ai llms",
                "javascript typescript react node.js express html css mongodb postgresql sql git docker restful apis next.js aws",
                "aws cloud computing terraform docker kubernetes networking security linux serverless microservices iam azure gcp",
                "networking firewalls linux bash cryptography wireshark penetration testing vulnerability assessment siem active directory cyber security",
                "python java go c# node.js express fastapi django postgresql mongodb redis sql restful apis system design",
                "python sql spark hadoop kafka etl data warehousing aws snowflake airflow data pipelines bigquery dbt",
                "swift kotlin java flutter react native dart ios development android development mobile design restful apis",
                "solidity ethereum rust smart contracts cryptography go web3 hardhat distributed systems",
                "product strategy agile scrum roadmapping user research wireframing jira market analysis communication product design",
                "figma adobe xd wireframing prototyping user research typography color theory visual design user flows interaction design",
                "selenium python java javascript qa automation cypress playwright manual testing api testing postman",
                "c c++ embedded systems microcontrollers rtos assembly hardware design firmware i2c spi uart"
            ],
            "Category": [
                "Data & AI", "Web Dev", "DevOps & Cloud", "Data & AI",
                "Web Dev", "DevOps & Cloud", "Cybersecurity", "Web Dev",
                "Data & AI", "Mobile Dev", "Web3 & Systems", "Management & Design",
                "Management & Design", "Testing & QA", "Hardware & Systems"
            ],
            "Avg_Salary": [
                "$120,000", "$95,000", "$115,000", "$145,000",
                "$110,000", "$140,000", "$115,000", "$115,000",
                "$125,000", "$110,000", "$135,000", "$120,000",
                "$95,000", "$95,000", "$115,000"
            ],
            "Growth_Rate": [
                "35%", "22%", "30%", "40%",
                "25%", "28%", "35%", "24%",
                "33%", "20%", "18%", "15%",
                "18%", "16%", "12%"
            ],
            "Description": [
                "Build statistical models, design a/b test criteria, and deploy core analytics systems.",
                "Build modern responsive front-end visual user interfaces and dynamic components.",
                "Automate application deployments, build robust pipelines, and configure hosting clouds.",
                "Design and train neural networks, implement transformer architectures, and scale vector search.",
                "Build end-to-end applications across both client layouts and backend API databases.",
                "Architect secure serverless resources, design multi-region clouds, and optimize IAM rules.",
                "Audit technical networks, run penetration scans, write encryption routines, and prevent hacks.",
                "Draft high-speed APIs, construct enterprise databases, and handle microservice systems.",
                "Construct pipelines, handle data migrations, build warehouses, and configure ETL workflows.",
                "Build Native iOS/Android modules, craft responsive interfaces, and deploy mobile apps.",
                "Write distributed smart contracts, handle decentralised cryptography, and build web3 systems.",
                "Analyze market trends, coordinate developers, and draft roadmap releases via Agile sprints.",
                "Create interactive mockups, design typographic layouts, and analyze user flows.",
                "Write automation suites, configure continuous tests, and audit API schemas.",
                "Configure firmware assemblies, integrate sensors on microcontrollers, and handle RTOS."
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv("skills.csv", index=False)
    df["Skills_lower"] = df["Skills"].str.lower()
    return df

df = load_career_data()

# ─────────────────────────────────────────────────────────
# RESUME PARSING FUNCTIONS
# ─────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file):
    try:
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_skills_from_text(text):
    text_lower = text.lower()
    extracted = []
    for skill in TECH_SKILLS:
        if len(skill) <= 3:
            pattern = rf"\b{re.escape(skill)}\b"
            if re.search(pattern, text_lower):
                extracted.append(skill.upper() if len(skill) > 2 else skill.title())
        else:
            if skill in text_lower:
                extracted.append(skill.title())
    return sorted(list(set(extracted)))

# ─────────────────────────────────────────────────────────
# ADVISOR SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────
st.sidebar.markdown(f"<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown(f"### ⚙️ PROFILE SETTINGS")

# Profile Selectboxes in Sidebar
st.sidebar.markdown('<p class="input-label">💼 Experience Level</p>', unsafe_allow_html=True)
experience = st.sidebar.selectbox("exp", ["Entry-level (0–2 yrs)", "Mid-level (2–5 yrs)", "Senior (5–10 yrs)", "Lead / Staff (10+ yrs)"], index=1, label_visibility="collapsed", key="experience")

st.sidebar.markdown('<p class="input-label">🎯 Primary Goal</p>', unsafe_allow_html=True)
goal = st.sidebar.selectbox("goal", ["Switch career", "Get promoted", "Upskill in current role", "Explore options", "Find highest paying role"], index=0, label_visibility="collapsed", key="goal")

st.sidebar.markdown(f"---")

# Custom Gemini key override option
st.sidebar.markdown(f"### 🔑 AI ADVISOR CO-PILOT")
custom_key = st.sidebar.text_input("Gemini API Key override", type="password", placeholder="Enter Google Gemini Key...", help="Leave blank to use the pre-configured default Gemini Key.")
active_key = custom_key if custom_key.strip() else API_KEY

if active_key:
    st.sidebar.markdown(f"<div style='font-size:11px; color:{badge_matched_text};'>✓ Advisor connected (gemini-2.5-flash)</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"<div style='font-size:11px; color:{badge_missing_text};'>⚠️ Advisor offline fallback active</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# MAIN HEADER (HERO)
# ─────────────────────────────────────────────────────────
col_hero, _ = st.columns([8, 1])
with col_hero:
    st.markdown(f'<span class="hero-badge">✦ Smart Career Engine</span>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="main-title" style="font-family:\'Plus Jakarta Sans\', sans-serif;">Find Your Perfect Career</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle" style="font-family:\'Plus Jakarta Sans\', sans-serif;">Enter your skills, or upload your resume. Our machine learning engine matches you across <strong style="color:{accent_color}">{len(df)} career paths</strong>, computes skills gap alignment, and compiles a personalized upskilling roadmap.</p>', unsafe_allow_html=True)

# Hero metric numbers
st.markdown(f"""
<div style="display:flex;gap:16px;margin-bottom:35px;flex-wrap:wrap;">
  <div style="flex:1;background:{card_bg};border:1px solid {card_border};border-radius:16px;padding:20px 14px;text-align:center;backdrop-filter:blur(8px);">
    <div style="font-size:26px;font-weight:800;background:{accent_gradient};-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{len(df)}</div>
    <div style="font-size:11px;color:{sub_text_color};letter-spacing:1.2px;text-transform:uppercase;margin-top:4px;font-weight:600;">Indexed Roles</div>
  </div>
  <div style="flex:1;background:{card_bg};border:1px solid {card_border};border-radius:16px;padding:20px 14px;text-align:center;backdrop-filter:blur(8px);">
    <div style="font-size:26px;font-weight:800;background:{accent_gradient};-webkit-background-clip:text;-webkit-text-fill-color:transparent;">120+</div>
    <div style="font-size:11px;color:{sub_text_color};letter-spacing:1.2px;text-transform:uppercase;margin-top:4px;font-weight:600;">Supported Skills</div>
  </div>
  <div style="flex:1;background:{card_bg};border:1px solid {card_border};border-radius:16px;padding:20px 14px;text-align:center;backdrop-filter:blur(8px);">
    <div style="font-size:26px;font-weight:800;background:{accent_gradient};-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Smart</div>
    <div style="font-size:11px;color:{sub_text_color};letter-spacing:1.2px;text-transform:uppercase;margin-top:4px;font-weight:600;">Gap Alignment</div>
  </div>
  <div style="flex:1;background:{card_bg};border:1px solid {card_border};border-radius:16px;padding:20px 14px;text-align:center;backdrop-filter:blur(8px);">
    <div style="font-size:26px;font-weight:800;background:{accent_gradient};-webkit-background-clip:text;-webkit-text-fill-color:transparent;">12 Weeks</div>
    <div style="font-size:11px;color:{sub_text_color};letter-spacing:1.2px;text-transform:uppercase;margin-top:4px;font-weight:600;">Action Roadmap</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Initialize button states prior to evaluation block to guarantee they are defined
clicked = False

# ─────────────────────────────────────────────────────────
# QUIZ AND INPUT ONBOARDING
# ─────────────────────────────────────────────────────────
if not st.session_state.quiz_active:
    st.markdown("""
    <div class="section-divider" style="display:flex;align-items:center;gap:14px;margin:10px 0 20px 0;">
      <div style="flex:1;height:1px;background:rgba(255,255,255,.06)"></div>
      <div style="font-size:16px;font-weight:700;color:#c9d6e8;font-family:Plus Jakarta Sans,sans-serif;white-space:nowrap">⚒️ Establish Your Technical Profile</div>
      <div style="flex:1;height:1px;background:rgba(255,255,255,.06)"></div>
    </div>
    """, unsafe_allow_html=True)

    col_profile_left, col_profile_right = st.columns([1, 1.2])

    with col_profile_left:
        with st.container(border=True):
            st.markdown('<p class="input-label">📄 Option A: Parse PDF/TXT Resume</p>', unsafe_allow_html=True)
            st.markdown("<div style='font-size:12px; color:#9ca3af; margin-bottom: 12px;'>Upload your resume. Our ML parser will instantly extract your technical skills!</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload PDF/TXT resume", type=["pdf", "txt"], label_visibility="collapsed", key="main_resume_uploader")
            
            # Process file upload
            if uploaded_file is not None:
                if uploaded_file.name != st.session_state.resume_name:
                    file_ext = uploaded_file.name.split(".")[-1].lower()
                    parsed_text = ""
                    if file_ext == "pdf":
                        with st.spinner("Decoding PDF resume..."):
                            parsed_text = extract_text_from_pdf(uploaded_file)
                    elif file_ext == "txt":
                        parsed_text = str(uploaded_file.read(), "utf-8", errors="ignore")
                    
                    st.session_state.parsed_resume_text = parsed_text
                    st.session_state.resume_name = uploaded_file.name
                    
                    if parsed_text.strip():
                        extracted = extract_skills_from_text(parsed_text)
                        st.session_state.extracted_skills = extracted
                        st.session_state.skills_input = ", ".join(extracted)
                        st.toast(f"✓ Found {len(extracted)} modern skills!", icon="🎉")
                        st.rerun()
                        
            if st.session_state.resume_name:
                st.markdown(f"""
                <div style='font-size:12.5px; color:#68d391; font-weight:600; padding: 10px 14px; border-radius: 12px; border: 1.2px solid rgba(104, 211, 145, 0.25); background: rgba(104, 211, 145, 0.08); margin-top:20px; display:flex; align-items:center; gap:8px;'>
                  <span>✓</span>
                  <div style='overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>Parsed: <b>{st.session_state.resume_name}</b> ({len(st.session_state.extracted_skills)} skills)</div>
                </div>
                """, unsafe_allow_html=True)

    with col_profile_right:
        with st.container(border=True):
            st.markdown('<p class="input-label">🛠️ Option B: Manual Input or Quiz</p>', unsafe_allow_html=True)
            
            # manual skills input
            user_input = st.text_input(
                "skills", 
                placeholder="e.g. Python, SQL, React, Git, Docker, Machine Learning",
                key="skills_input", 
                label_visibility="collapsed"
            )
            
            # Quiz Trigger button
            col_quiz_lbl, col_quiz_btn = st.columns([1.5, 1])
            with col_quiz_lbl:
                st.markdown("<div style='font-size:12px; color:#9ca3af; margin-top: 8px;'>Don't know what skills to input?</div>", unsafe_allow_html=True)
            with col_quiz_btn:
                if st.button("📝 Discover Skills", key="start_quiz_btn"):
                    st.session_state.quiz_active = True
                    st.session_state.quiz_step = 0
                    st.session_state.quiz_answers = {}
                    st.rerun()
                    
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            # Quick Add Tags grid
            st.markdown(f"<div style='font-size:11px; color:#9ca3af; margin-bottom: 6px; font-weight:700; letter-spacing:0.5px;'>QUICK ADD POPULAR SKILLS:</div>", unsafe_allow_html=True)
            all_tags = ["Python", "JavaScript", "SQL", "React", "Docker", "Machine Learning", "AWS", "Kubernetes", "Git", "Go"]
            
            cols = st.columns(5)
            for idx, skill in enumerate(all_tags):
                with cols[idx % 5]:
                    st.button(skill, key=f"tag_{skill}", on_click=add_skill, args=(skill,))

    with st.container(border=True):
        col_cat_flt, col_discover = st.columns([1.2, 1])
        with col_cat_flt:
            categories = ["All"] + sorted(df["Category"].unique().tolist())
            category_filter = st.selectbox("cat", categories, label_visibility="collapsed", key="cat_filter")
            
        with col_discover:
            clicked = st.button("🚀  Discover My Career Matches", type="secondary")
else:
    with st.container(border=True):
        # Render interactive skill-discovery onboarding quiz
        st.markdown(f"<h3 style='color:{accent_color}; font-weight:800; margin:0;'>📝 Skill Discovery Assistant</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:13.5px; color:{sub_text_color}; margin-bottom: 22px;'>Answer these quick multiple-choice questions to map out your core technical skills automatically.</p>", unsafe_allow_html=True)
        
        quiz_questions = [
            {
                "question": "Which software layer or engineering discipline excites you the most?",
                "options": [
                    "Artificial Intelligence, Neural networks, NLP algorithms, and Big Data scaling",
                    "Sleek client side views, visual grids, color theories, and responsive web systems",
                    "Multi-threaded API systems, scalable databases, query compilers, and microservices",
                    "Docker virtualization, cloud servers, telemetry dashboards, CI/CD orchestration",
                    "Sensor integrations, firmware assembly codes, secure firewalls, and system packets"
                ],
                "skills": [
                    ["python", "machine learning", "pandas", "numpy", "sql", "statistics"],
                    ["html", "css", "javascript", "react", "figma", "web design"],
                    ["node.js", "express", "postgresql", "mongodb", "fastapi", "sql", "redis"],
                    ["docker", "kubernetes", "aws", "terraform", "ci/cd", "linux", "bash"],
                    ["c", "c++", "embedded systems", "linux", "networking", "security", "firmware"]
                ]
            },
            {
                "question": "Which programming language families represent your primary comfort zone?",
                "options": [
                    "Python / R (AI engineering, numeric computation, scripting models)",
                    "JavaScript / TypeScript (Modern visual clients, React ecosystems, Node layers)",
                    "Java / Go / C# (Enterprise scales, microservice APIs, static types)",
                    "C / C++ / Rust / Solidity (Firmware pipelines, low-level compilers, web3 networks)",
                    "Bash / PowerShell / Shell Scripting (System automation, configs, pipelines)"
                ],
                "skills": [
                    ["python", "pandas", "numpy", "scikit-learn", "r"],
                    ["javascript", "typescript", "react", "node.js", "css", "html"],
                    ["java", "go", "c#", "restful apis", "system design"],
                    ["c", "c++", "rust", "embedded systems", "solidity", "smart contracts"],
                    ["linux", "bash", "shell scripting", "git", "github"]
                ]
            },
            {
                "question": "How do you prefer to manage database layers or record schemas?",
                "options": [
                    "Writing strict relational SQL queries, table indices, and transaction models",
                    "Working with JSON documents, key-values, and schema-less NoSQL platforms (MongoDB)",
                    "Engineering distributed message flows, batch streams, and ETL mappings (Spark, Kafka)",
                    "Designing secure multi-region clouds, network access rules, and redundancy plans",
                    "I focus strictly on layouts, wireframes, and design components instead of databases"
                ],
                "skills": [
                    ["sql", "postgresql", "mysql"],
                    ["mongodb", "redis", "nosql"],
                    ["spark", "kafka", "etl", "data warehousing", "bigquery"],
                    ["aws", "cloud computing", "iam", "gcp", "azure"],
                    ["figma", "wireframing", "user research", "product design"]
                ]
            },
            {
                "question": "What is your preference regarding software hosting, building, and deployments?",
                "options": [
                    "I configure multi-container namespaces, write YAML structures, and compile pipeline workflows",
                    "I prefer standard cloud deployments on platforms like Amazon AWS, Azure, or GCP",
                    "I build static server-rendered visual pages using React frameworks (Next.js, TailwindCSS)",
                    "I assemble electrical circuitry, integrate physical interfaces, and configure real-time OS boards",
                    "I prefer prioritizing team feature backlogs, coordinating sprints, and roadmapping product launches"
                ],
                "skills": [
                    ["docker", "kubernetes", "ci/cd", "github actions", "git", "jenkins"],
                    ["aws", "gcp", "azure", "cloud computing"],
                    ["next.js", "tailwindcss", "html", "css", "react"],
                    ["embedded systems", "microcontrollers", "firmware", "rtos"],
                    ["product strategy", "agile", "scrum", "roadmapping", "jira"]
                ]
            },
            {
                "question": "Which of these workflow toolings feels the most natural to you?",
                "options": [
                    "Creating interface layouts, visual prototypes, and component libraries inside Figma",
                    "Tuning tensor layers and learning networks using PyTorch or TensorFlow environments",
                    "Writing automated validation suites using systems like Selenium, Cypress, or Playwright",
                    "Drafting secure decentralised ledger smart contracts with Solidity and Hardhat",
                    "Managing team backlogs in Jira and establishing business product directions"
                ],
                "skills": [
                    ["figma", "adobe xd", "wireframing", "visual design", "typography"],
                    ["pytorch", "tensorflow", "deep learning", "nlp", "computer vision"],
                    ["selenium", "qa automation", "cypress", "playwright", "api testing"],
                    ["solidity", "smart contracts", "cryptography", "web3", "hardhat"],
                    ["jira", "scrum", "product strategy", "communication"]
                ]
            }
        ]
        
        current_step = st.session_state.quiz_step
        q_data = quiz_questions[current_step]
        
        st.markdown(f"<div style='font-size:11.5px; color:{accent_color}; font-weight:700;'>STEP {current_step+1} OF {len(quiz_questions)}</div>", unsafe_allow_html=True)
        st.progress((current_step) / len(quiz_questions))
        
        st.markdown(f"<p style='font-size:16px; font-weight:700; margin:16px 0 10px 0;'>{q_data['question']}</p>", unsafe_allow_html=True)
        
        selected_val = st.radio("opts", q_data["options"], key=f"quiz_rad_{current_step}", label_visibility="collapsed")
        
        st.markdown(f"<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        col_quiz_btn1, col_quiz_btn2 = st.columns([1, 1])
        
        with col_quiz_btn1:
            if st.button("➔ Next Question" if current_step < len(quiz_questions) - 1 else "🏁 Compile Skills", type="secondary", key="quiz_nxt_btn"):
                opt_idx = q_data["options"].index(selected_val)
                st.session_state.quiz_answers[current_step] = q_data["skills"][opt_idx]
                
                if current_step < len(quiz_questions) - 1:
                    st.session_state.quiz_step += 1
                    st.rerun()
                else:
                    all_skills = []
                    for _, lst in st.session_state.quiz_answers.items():
                        all_skills.extend(lst)
                    unique_skills = sorted(list(set(all_skills)))
                    st.session_state.skills_input = ", ".join(unique_skills)
                    st.session_state.quiz_active = False
                    st.session_state.quiz_step = 0
                    st.success("🎉 Technical profile compiled! Loaded skills into the workspace.")
                    time.sleep(1.2)
                    st.rerun()
                    
        with col_quiz_btn2:
            if st.button("❌ Exit Assistant", key="quiz_cancel_btn"):
                st.session_state.quiz_active = False
                st.rerun()

# ─────────────────────────────────────────────────────────
# ENGINE MATCHING CALCULATIONS
# ─────────────────────────────────────────────────────────
def get_matches(user_skills_raw, category="All", top_n=5):
    clean = user_skills_raw.lower().replace(",", " ")
    fdf = df.copy()
    if category != "All":
        fdf = fdf[fdf["Category"] == category]
    if fdf.empty:
        fdf = df.copy()
    corpus = fdf["Skills_lower"].tolist() + [clean]
    vec = TfidfVectorizer(token_pattern=r'(?u)\b[\w\-\.\#\+]{2,}\b')
    mat = vec.fit_transform(corpus)
    sim = cosine_similarity(mat[-1], mat[:-1])[0]
    fdf = fdf.copy()
    fdf["score"] = sim
    return fdf.sort_values("score", ascending=False).head(top_n)

# ─────────────────────────────────────────────────────────
# GEMINI PERSONALIZED AI SYLLABUS AGENT
# ─────────────────────────────────────────────────────────
def build_prompt(user_skills, role, role_skills, experience, goal):
    return f"""You are an expert tech career coach giving personalized advice.

User Details:
- Skills they currently have: {user_skills}
- Experience Level: {experience}
- Transition Goal: {goal}
- Target Career Role: {role}
- Role requires: {role_skills}

Provide a completely filled personalized tech assessment roadmap.
You MUST reply ONLY with a valid JSON block containing precisely these keys. No markdown backticks, no text before or after, just pure JSON:
{{
  "summary": "2-3 sentence career fit assessment based on their experience level and transition goal.",
  "skill_gaps": ["gap1","gap2","gap3","gap4","gap5"],
  "user_has": ["skill1","skill2","skill3"],
  "roadmap": [
    {{"week":"1-2","title":"Short title","task":"Exactly what to build or read","priority":"high"}},
    {{"week":"3-4","title":"Short title","task":"Exactly what to build or read","priority":"high"}},
    {{"week":"5-6","title":"Short title","task":"Exactly what to build or read","priority":"medium"}},
    {{"week":"7-8","title":"Short title","task":"Exactly what to build or read","priority":"medium"}},
    {{"week":"9-10","title":"Short title","task":"Exactly what to build or read","priority":"low"}},
    {{"week":"11-12","title":"Short title","task":"Exactly what to build or read","priority":"low"}}
  ],
  "next_steps": ["action1","action2","action3"],
  "salary_insight": "one sentence salary insight corresponding to the experience level and target role"
}}"""

def generate_local_fallback_insights(user_skills, role, role_skills, experience, goal, error_msg=""):
    user_skills_set = set([s.strip().lower() for s in user_skills.split(",") if s.strip()])
    role_skills_set = set([s.strip().lower() for s in role_skills.split(",") if s.strip()])
    
    user_has = sorted(list(user_skills_set.intersection(role_skills_set)))
    skill_gaps = sorted(list(role_skills_set.difference(user_skills_set)))
    
    # Prettify labels
    user_has = [s.title() for s in user_has] if user_has else ["No matches found yet"]
    skill_gaps = [s.title() for s in skill_gaps] if skill_gaps else ["None! Ready to scale"]
    
    roadmap = [
        {"week": "1-2", "title": f"Core Foundations: {skill_gaps[0]}", "task": f"Learn variables, parameters, and basic structure files for {skill_gaps[0]}", "priority": "high"},
        {"week": "3-4", "title": f"Integration: {skill_gaps[1] if len(skill_gaps)>1 else 'Tooling Scaling'}", "task": f"Build practical modules connecting {skill_gaps[1] if len(skill_gaps)>1 else 'core frameworks'} to your current stack.", "priority": "high"},
        {"week": "5-6", "title": "Connecting Database Layers", "task": "Configure structured repositories, check schemas, and connect robust data storage systems.", "priority": "medium"},
        {"week": "7-8", "title": "Telemetry & Pipelines Setup", "task": "Set up remote containers, construct docker build files, and manage automated pipeline tests.", "priority": "medium"},
        {"week": "9-10", "title": "System Scalability Reviews", "task": "Study cache replications, query structures, thread pooling, and multi-threaded scaling mechanisms.", "priority": "low"},
        {"week": "11-12", "title": "Portfolio Launch & Revision", "task": "Publish showcase pipelines to Git, compile detailed documentation, and structure your resume.", "priority": "low"}
    ]
    
    next_steps = [
        f"Initialize a dedicated GitHub repository for your {role} career transition projects.",
        f"Schedule 1.5 hours daily studying {skill_gaps[0]}.",
        "Re-architect an existing codebase to implement structured scaling standards."
    ]
    
    # Pre-coded growth & salary insights based on experience levels
    if "Entry" in experience:
        salary_insight = f"Entry level {role} roles generally command starting salaries of $75,000 to $95,000 in major tech ecosystems."
    elif "Mid" in experience:
        salary_insight = f"Mid-level {role} candidates are highly sought after, with average bases of $105,000 to $125,000 plus benefits."
    else:
        salary_insight = f"Senior/Lead {role} specialists easily secure premium baselines of $145,000 to $185,000 with equity opportunities."
        
    summary = f"Based on your manual profile, you have an initial compatibility alignment with the {role} career model. Capitalizing on your {', '.join(user_has[:2])} strengths provides a great baseline."
    
    if error_msg:
        summary += " [Offline Analysis Mode Active]"
        
    return {
        "summary": summary,
        "skill_gaps": skill_gaps[:5],
        "user_has": user_has[:5],
        "roadmap": roadmap,
        "next_steps": next_steps,
        "salary_insight": salary_insight
    }

def get_ai_insights(user_skills, role, role_skills, experience, goal):
    if not active_key:
        return generate_local_fallback_insights(user_skills, role, role_skills, experience, goal, "No API key configured.")
    try:
        genai.configure(api_key=active_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = build_prompt(user_skills, role, role_skills, experience, goal)
        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        return generate_local_fallback_insights(user_skills, role, role_skills, experience, goal, str(e))

# ─────────────────────────────────────────────────────────
# MAIN PROCESSING & RESULTS PLOTTING
# ─────────────────────────────────────────────────────────
if clicked or st.session_state.skills_input.strip() != "":
    if not st.session_state.skills_input.strip():
        st.warning("⚠️ Please provide at least one skill or upload your resume in the sidebar to discover career matches.")
    else:
        # 1. Matching
        cat_val = st.session_state.get("cat_filter", "All")
        top = get_matches(st.session_state.skills_input, cat_val, top_n=5)
        
        # Section Divider
        st.markdown("""
        <div class="section-divider" style="display:flex;align-items:center;gap:14px;margin:32px 0 24px 0;">
          <div style="flex:1;height:1px;background:rgba(255,255,255,.06)"></div>
          <div style="font-size:16px;font-weight:700;color:#c9d6e8;font-family:Plus Jakarta Sans,sans-serif;white-space:nowrap">🎯 Your Technical Matching Leaderboard</div>
          <div style="flex:1;height:1px;background:rgba(255,255,255,.06)"></div>
        </div>
        """, unsafe_allow_html=True)
        
        col_leader, col_radar = st.columns([1.1, 0.9])
        
        with col_leader:
            with st.container(border=True):
                st.markdown('<p class="input-label">📊 Cosine Match Compatibilities</p>', unsafe_allow_html=True)
                
                # Interactive matching scores plot
                chart_df = top.copy()
                chart_df["Match Score (%)"] = chart_df["score"] * 100
                chart_df = chart_df.sort_values(by="Match Score (%)", ascending=True)
                
                fig_bar = px.bar(
                    chart_df,
                    x="Match Score (%)",
                    y="Role",
                    orientation="h",
                    color="Match Score (%)",
                    color_continuous_scale=["#7c3aed", "#db2777"] if theme_choice == "Orchid Light (Clean)" else ["#a855f7", "#ec4899", "#f43f5e"],
                    labels={"Match Score (%)": "Alignment Score (%)", "Role": "Career role"},
                    text_auto='.1f'
                )
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color=text_color,
                    coloraxis_showscale=False,
                    height=240,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(showgrid=True, gridcolor=grid_color, range=[0, 100]),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_radar:
            with st.container(border=True):
                st.markdown('<p class="input-label">🕸️ Profile Domain Strengths</p>', unsafe_allow_html=True)
                
                # Map input skills into domain scores
                input_set = set([s.strip().lower() for s in st.session_state.skills_input.split(",") if s.strip()])
                domain_scores = {}
                for dom, sks in DOMAINS.items():
                    domain_matches = [s for s in input_set if s in sks]
                    domain_scores[dom] = len(domain_matches)
                    
                categories = list(DOMAINS.keys())
                r_vals = [domain_scores[c] for c in categories]
                
                # Radar chart
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_vals,
                    theta=categories,
                    fill='toself',
                    fillcolor='rgba(168, 85, 247, 0.15)',
                    line=dict(color=accent_color, width=2),
                    marker=dict(size=6, color=accent_color)
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max(max(r_vals), 3)], gridcolor=grid_color, tickfont=dict(color=sub_text_color, size=8)),
                        angularaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color, size=8))
                    ),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=240,
                    margin=dict(l=50, r=50, t=10, b=10)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
        # Top Career Matches Title
        st.markdown("""
        <div class="section-divider" style="display:flex;align-items:center;gap:14px;margin:24px 0 18px 0;">
          <div style="flex:1;height:1px;background:rgba(255,255,255,.06)"></div>
          <div style="font-size:16px;font-weight:700;color:#c9d6e8;font-family:Plus Jakarta Sans,sans-serif;white-space:nowrap">🥇 Explore Recommended Careers & Custom Roadmaps</div>
          <div style="flex:1;height:1px;background:rgba(255,255,255,.06)"></div>
        </div>
        """, unsafe_allow_html=True)
        
        medals = ["🥇", "🥈", "🥉", "④", "⑤"]
        for idx, (_, row) in enumerate(top.iterrows()):
            pct = row["score"] * 100
            rank = idx + 1
            bar_width = min(pct, 100)
            
            best_badge = ""
            if rank == 1:
                best_badge = f'<div style="display:inline-block;background:{accent_gradient};color:{primary_btn_text};font-size:10px;font-weight:800;letter-spacing:1.2px;padding:4px 12px;border-radius:20px;margin-bottom:10px;text-transform:uppercase;font-family:Plus Jakarta Sans,sans-serif;">⭐ BEST PROFILE FIT</div>'
                
            st.markdown(f"""
            <div class="premium-card" style="position:relative; overflow:hidden;">
              <div style="position:absolute; top:0; left:0; width:4px; height:100%; background:{accent_gradient}; border-radius:4px 0 0 4px;"></div>
              {best_badge}
              <div class="card-rank">{medals[idx]} RANK #{rank} &nbsp;·&nbsp; {row['Category']}</div>
              <div class="card-role" style="font-family:Plus Jakarta Sans,sans-serif;">{row['Role']}</div>
              <div style="font-size:13.5px; color:{sub_text_color}; line-height:1.55; margin-bottom:14px; font-weight:300;">{row['Description']}</div>
              
              <div style="display:flex; gap:28px; margin-bottom:16px; flex-wrap:wrap;">
                <div>
                  <div style="font-size:11px; color:{sub_text_color}; text-transform:uppercase; letter-spacing:0.8px; font-weight:600;">Avg Salary</div>
                  <div style="font-size:15px; font-weight:800; color:{accent_color};">{row['Avg_Salary']}</div>
                </div>
                <div>
                  <div style="font-size:11px; color:{sub_text_color}; text-transform:uppercase; letter-spacing:0.8px; font-weight:600;">Growth Rate</div>
                  <div style="font-size:15px; font-weight:800; color:{accent_color};">{row['Growth_Rate']}</div>
                </div>
                <div>
                  <div style="font-size:11px; color:{sub_text_color}; text-transform:uppercase; letter-spacing:0.8px; font-weight:600;">Match Score</div>
                  <div style="font-size:15px; font-weight:800; color:{accent_color};">{pct:.1f}%</div>
                </div>
              </div>
              
              <div style="height:6px; background:rgba(255,255,255,0.06); border-radius:6px; overflow:hidden;">
                <div style="height:6px; width:{bar_width:.1f}%; border-radius:6px; background:{accent_gradient};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Deep insights syllabus for top 3
            if rank <= 3:
                with st.expander(f"📊 Get Personalized AI Insights & Week-by-Week Learning Roadmap for {row['Role']}", expanded=(rank == 1)):
                    role_name = row["Role"]
                    
                    # Caching system to prevent synchronous API calling on load
                    has_cached = role_name in st.session_state.ai_insights_cache
                    
                    if not has_cached:
                        st.markdown(f"""
                        <div style='font-size: 13.5px; color:{sub_text_color}; margin-bottom: 20px; font-weight: 300; line-height: 1.5;'>
                          Connect with Google Gemini to generate a highly detailed, personalized **12-week study syllabus** and **action plan** tailored for <b>{role_name}</b> based on your experience level and transition goals.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"⚡ Generate Personalized Roadmap for {role_name}", key=f"gen_ai_btn_{role_name}", type="primary"):
                            with st.spinner("Connecting with Advisor Co-Pilot..."):
                                insights = get_ai_insights(
                                    st.session_state.skills_input, 
                                    role_name, 
                                    row["Skills"], 
                                    st.session_state.experience, 
                                    st.session_state.goal
                                )
                                st.session_state.ai_insights_cache[role_name] = insights
                                st.rerun()
                    else:
                        insights = st.session_state.ai_insights_cache[role_name]
                        if "error" in insights:
                            st.error(f"❌ {insights['error']}")
                            if st.button("Retry AI Generation", key=f"retry_{role_name}"):
                                del st.session_state.ai_insights_cache[role_name]
                                st.rerun()
                        else:
                            # 1. Assessment Banner
                            st.markdown(f"""
                            <div class="premium-card" style="background:rgba(168, 85, 247, 0.04); border: 1.5px solid {card_border}; margin-bottom:20px;">
                                <div style="font-size:11px; font-weight:700; color:{accent_color}; letter-spacing:1.8px; text-transform:uppercase; margin-bottom:8px;">📋 CAREER PROFILE FIT SUMMARY</div>
                                <div style="font-size:14px; color:{text_color}; line-height:1.6; font-weight:300; margin-bottom:10px;">{insights.get('summary', '')}</div>
                                <div style="font-size:13px; color:{accent_color}; font-weight:600;">💡 Base compensation insights: {insights.get('salary_insight', '')}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 2. Possessed vs Missing
                            col_pos, col_mis = st.columns(2)
                            with col_pos:
                                st.markdown(f"<div style='font-size:11.5px; font-weight:700; color:{badge_matched_text}; letter-spacing:1px; margin-bottom:8px; text-transform:uppercase;'>✅ Skills Possessed ({len(insights.get('user_has', []))})</div>", unsafe_allow_html=True)
                                has_skills = insights.get("user_has", [])
                                if has_skills:
                                    badges_has = "".join([f'<span class="badge-have">✓ {s.upper()}</span>' for s in has_skills])
                                    st.markdown(f"<div>{badges_has}</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<div style='font-size:13px; color:{sub_text_color}; font-weight:300;'>No technical matches detected yet.</div>", unsafe_allow_html=True)
                            
                            with col_mis:
                                st.markdown(f"<div style='font-size:11.5px; font-weight:700; color:{badge_missing_text}; letter-spacing:1px; margin-bottom:8px; text-transform:uppercase;'>🛠️ Technical Skill Gaps ({len(insights.get('skill_gaps', []))})</div>", unsafe_allow_html=True)
                                need_skills = insights.get("skill_gaps", [])
                                if need_skills:
                                    badges_need = "".join([f'<span class="badge-need">⚠️ {s.upper()}</span>' for s in need_skills])
                                    st.markdown(f"<div>{badges_need}</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<div style='font-size:13px; color:{badge_matched_text}; font-weight:700;'>Perfect match! You have all core skills for this role.</div>", unsafe_allow_html=True)
                                    
                            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                            
                            # 3. 12-Week Roadmap Timeline
                            st.markdown("""
                            <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
                              <div style="flex:1;height:1px;background:rgba(255,255,255,.05)"></div>
                              <div style="font-size:13.5px;font-weight:700;color:#c9d6e8;font-family:Plus Jakarta Sans,sans-serif;text-transform:uppercase;letter-spacing:1px;">📅 Your 12-Week Adaptive Curriculum</div>
                              <div style="flex:1;height:1px;background:rgba(255,255,255,.05)"></div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            priority_colors = {"high": "#fc814a", "medium": "#f6ad55", "low": "#a855f7"}
                            priority_bg = {"high": "rgba(252,129,74,.12)", "medium": "rgba(246,173,85,.10)", "low": "rgba(168,85,247,.10)"}
                            
                            for step in insights.get("roadmap", []):
                                pri = step.get("priority", "low").lower()
                                p_color = priority_colors.get(pri, "#a855f7")
                                p_bg = priority_bg.get(pri, "rgba(168,85,247,.10)")
                                
                                st.markdown(f"""
                                <div class="timeline-card">
                                  <div class="timeline-badge">
                                    <div class="timeline-wk">WK</div>
                                    <div class="timeline-num">{step.get('week', '')}</div>
                                  </div>
                                  <div class="timeline-content">
                                    <div class="timeline-title">
                                      {step.get('title', '')}
                                      <span class="priority-lbl" style="background:{p_bg}; color:{p_color}; border:1px solid {p_color}44;">{pri} priority</span>
                                    </div>
                                    <div class="timeline-task">{step.get('task', '')}</div>
                                  </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            # 4. Next Steps & Actions
                            st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
                            st.markdown("""
                            <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;">
                              <div style="flex:1;height:1px;background:rgba(255,255,255,.05)"></div>
                              <div style="font-size:13.5px;font-weight:700;color:#c9d6e8;font-family:Plus Jakarta Sans,sans-serif;text-transform:uppercase;letter-spacing:1px;">⚡ Critical Next Steps</div>
                              <div style="flex:1;height:1px;background:rgba(255,255,255,.05)"></div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for i, action in enumerate(insights.get("next_steps", []), 1):
                                st.markdown(f"""
                                <div style="display:flex;align-items:center;gap:12px;padding:12px 16px;border-radius:12px;margin-bottom:8px;background:{stat_box_bg};border:1px solid {card_border};backdrop-filter:blur(4px);">
                                  <div style="min-width:32px;height:32px;border-radius:50%;background:{accent_gradient};display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:800;color:{primary_btn_text};">{i}</div>
                                  <div style="font-size:13.5px;color:{text_color};font-weight:400;">{action}</div>
                                </div>
                                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  Built with <span style="color:{accent_color};">♥</span> · CareerAI Smart Recommendation Engine · Powered by TF-IDF Similarity+Cosine Similarity 
</div>
""", unsafe_allow_html=True)
