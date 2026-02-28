# ============================================================
#  EMAN SQL AGENT  —  Redesigned UI by Claude
# ============================================================
import os
import json
import sqlite3
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

try:
    import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_ROWS = 20

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Exo+2:wght@300;400;600&display=swap');

:root {
    --bg-main:      #0d1117;
    --bg-card:      #161b22;
    --accent-blue:  #2196f3;
    --accent-gold:  #f0b429;
    --accent-cyan:  #00e5ff;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --border:       #30363d;
    --success:      #3fb950;
    --error:        #f85149;
    --glow-gold:    0 0 18px rgba(240,180,41,0.45);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Exo 2', sans-serif !important;
}
#MainMenu, footer { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #101823 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

.eman-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #2196f3, #00e5ff, #f0b429);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
}
.eman-subtitle {
    font-family: 'Exo 2', sans-serif;
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-top: 6px;
    margin-bottom: 10px;
}
.eman-features { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.feat-pill {
    display: inline-block;
    background: rgba(33,150,243,0.1);
    border: 1px solid rgba(33,150,243,0.3);
    color: #93c5fd;
    font-family: 'Exo 2', sans-serif;
    font-size: 0.78rem;
    padding: 4px 12px;
    border-radius: 20px;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-blue);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    transition: border-color 0.3s;
}
.metric-card:hover { border-left-color: var(--accent-gold); }
.metric-label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-family: 'Orbitron', monospace; font-size: 1.3rem; color: var(--accent-cyan); font-weight: 700; }

.sidebar-section {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    color: var(--accent-gold);
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
    margin: 18px 0 10px 0;
}

.hist-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 11px;
    margin-bottom: 5px;
    font-size: 0.8rem;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.think-box {
    background: rgba(240,180,41,0.07);
    border-left: 3px solid var(--accent-gold);
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #c9a227;
    font-style: italic;
}

pre, code {
    background: #0d1f33 !important;
    color: var(--accent-cyan) !important;
    border: 1px solid rgba(33,150,243,0.3) !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1a2744 0%, #1e3a5f 100%) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid rgba(33,150,243,0.4) !important;
    border-radius: 8px !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.82rem !important;
    transition: all 0.25s !important;
    padding: 6px 12px !important;
}
.stButton > button:hover {
    border-color: var(--accent-gold) !important;
    color: var(--accent-gold) !important;
    box-shadow: var(--glow-gold) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px !important; }
[data-testid="stChatInput"] textarea {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--accent-blue) !important;
    border-radius: 12px !important;
    font-family: 'Exo 2', sans-serif !important;
}
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
}
hr { border-color: var(--border) !important; }
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--accent-blue) !important;
    border-radius: 10px !important;
}
input[type="text"], input[type="password"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
.schema-box {
    background: #0d1f33;
    border: 1px solid rgba(33,150,243,0.3);
    border-radius: 10px;
    padding: 14px;
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    color: #7dd3fc;
    white-space: pre;
    overflow-x: auto;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-online { background: rgba(63,185,80,0.15); color: var(--success); border: 1px solid rgba(63,185,80,0.35); }
.badge-offline { background: rgba(248,81,73,0.15); color: var(--error); border: 1px solid rgba(248,81,73,0.35); }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
</style>
"""

# ─────────────────────────────────────────────────────────
#  LLM
# ─────────────────────────────────────────────────────────
@st.cache_resource
def get_llm(groq_api_key: str | None):
    key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Groq API key not provided. Set in sidebar or env GROQ_API_KEY")
    return ChatGroq(model=DEFAULT_MODEL, temperature=0, streaming=False, api_key=key)


# ─────────────────────────────────────────────────────────
#  Database helpers
# ─────────────────────────────────────────────────────────
def save_uploaded_db(uploaded_file) -> str:
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        return tmp.name

def get_db_path(uploaded_file) -> str | None:
    if uploaded_file is not None:
        return save_uploaded_db(uploaded_file)
    default_db = Path(__file__).parent / "student.db"
    if default_db.exists():
        return str(default_db)
    return None

def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def get_schema(conn: sqlite3.Connection) -> dict:
    schema = {}
    cur = conn.cursor()
    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    ).fetchall()
    for (table_name,) in tables:
        cols = cur.execute(f"PRAGMA table_info({table_name});").fetchall()
        schema[table_name] = [c[1] for c in cols]
    return schema

def schema_to_text(schema: dict) -> str:
    lines = []
    for table, cols in schema.items():
        preview = ", ".join(cols[:10])
        extra = " ..." if len(cols) > 10 else ""
        lines.append(f"  {table}({preview}{extra})")
    return "\n".join(lines)

def get_db_stats(conn: sqlite3.Connection, schema: dict) -> dict:
    stats = {}
    cur = conn.cursor()
    for table in schema:
        try:
            count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = count
        except Exception:
            stats[table] = "?"
    return stats


# ─────────────────────────────────────────────────────────
#  Auto-generate example questions from schema
# ─────────────────────────────────────────────────────────
def generate_example_questions(llm: ChatGroq, schema_text: str) -> list[str]:
    system = SystemMessage(content=(
        "You are a SQL expert. Based on the given database schema, "
        "generate exactly 5 short, practical example questions a user might ask. "
        "Questions should be specific to the actual tables and columns in the schema. "
        "Return ONLY a JSON array of 5 strings, nothing else. No explanation, no markdown. "
        'Example format: ["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]'
    ))
    user = HumanMessage(content=f"SCHEMA:\n{schema_text}\n\nReturn 5 example questions as a JSON array.")
    try:
        resp = llm.invoke([system, user])
        text = resp.content.strip()
        start = text.index("[")
        end = text.rindex("]") + 1
        questions = json.loads(text[start:end])
        return [str(q) for q in questions[:5]]
    except Exception:
        return [
            "Show all tables and their row counts",
            "Show first 10 rows of the main table",
            "Count total records in each table",
            "Find any duplicate entries",
            "Show all column names in each table",
        ]


# ─────────────────────────────────────────────────────────
#  Agent logic
# ─────────────────────────────────────────────────────────
def ask_llm_for_sql(llm: ChatGroq, question: str, schema_text: str) -> dict:
    system = SystemMessage(content=(
        "You are 'Eman SQL Agent', a smart SQL expert for SQLite databases.\n"
        "You MUST use ONLY the tables and columns listed in SCHEMA below.\n"
        "Write ONLY safe SELECT queries (no INSERT/UPDATE/DELETE/DROP/PRAGMA).\n"
        "If the question is vague, make a reasonable assumption and mention it in 'thinking'.\n"
        "ALWAYS add LIMIT 20 if user does not specify one.\n"
        "Return STRICT JSON with keys: sql, thinking, followups.\n"
        "followups = list of 3 smart follow-up questions.\n"
        f"SCHEMA:\n{schema_text}"
    ))
    user = HumanMessage(content=(
        f"User question: {question}\n\n"
        "Reply ONLY in JSON:\n"
        '{"sql":"...","thinking":"...","followups":["...","...","..."]}'
    ))
    resp = llm.invoke([system, user])
    text = resp.content.strip()
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
    except Exception:
        data = {
            "sql": "SELECT 'Sorry, could not generate SQL' AS error",
            "thinking": "Failed to follow JSON format.",
            "followups": ["Try simpler question", "What tables are available?", "Show all columns"]
        }
    return data


def run_sql(conn: sqlite3.Connection, sql: str) -> pd.DataFrame | str:
    sql_clean = sql.strip().rstrip(";")
    if not sql_clean.lower().startswith("select"):
        return "🚫 Blocked: Only SELECT queries are allowed."
    if "limit" not in sql_clean.lower():
        sql_to_run = f"{sql_clean} LIMIT {MAX_ROWS}"
    else:
        sql_to_run = sql_clean
    try:
        return pd.read_sql_query(sql_to_run, conn)
    except Exception as e:
        return f"SQL Error: {e}"


def build_final_answer(llm: ChatGroq, question: str, sql: str, result) -> str:
    if isinstance(result, pd.DataFrame):
        if result.empty:
            result_text = "Query returned 0 rows."
        else:
            preview = result.head(min(5, len(result)))
            result_text = "Preview (up to 5 rows):\n"
            result_text += (preview.to_markdown(index=False)
                            if HAS_TABULATE else preview.to_string(index=False))
    else:
        result_text = str(result)

    system = SystemMessage(content=(
        "You are 'Eman SQL Agent', an AI data assistant.\n"
        "Explain the SQL result clearly and concisely.\n"
        "If there was an error, explain it gently and suggest a fix.\n"
        "Keep it professional but friendly. End with one short encouraging line."
    ))
    user = HumanMessage(content=(
        f"User question: {question}\nSQL used:\n{sql}\n\nResult:\n{result_text}"
    ))
    resp = llm.invoke([system, user])
    return resp.content.strip()


# ─────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Eman SQL Agent",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Session state init ──
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "query_count" not in st.session_state:
        st.session_state["query_count"] = 0
    if "pending_question" not in st.session_state:
        st.session_state["pending_question"] = None
    if "example_tips" not in st.session_state:
        st.session_state["example_tips"] = []
    if "last_schema_key" not in st.session_state:
        st.session_state["last_schema_key"] = None
    if "last_followups" not in st.session_state:
        st.session_state["last_followups"] = []

    # ══════════════════════════════════════════════════════
    #  SIDEBAR
    # ══════════════════════════════════════════════════════
    with st.sidebar:

        st.markdown("""
        <div style='text-align:center; padding: 10px 0 6px 0;'>
            <div style='font-family: Orbitron, monospace; font-size:1.3rem; font-weight:700;
                        background: linear-gradient(90deg,#2196f3,#00e5ff,#f0b429);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text; letter-spacing:3px;'>
                💎 Ask. Analyze. Answer.
            </div>
        </div>
        <hr style='border-color:#30363d; margin:8px 0;'/>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">📂 Database</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload SQLite .db", type=["db", "sqlite"], label_visibility="collapsed")
        st.caption("Upload `.db` file or keep `student.db` in script folder.")

        st.markdown('<div class="sidebar-section">🔑 Groq API Key</div>', unsafe_allow_html=True)
        key_input = st.text_input("GROQ_API_KEY", type="password",
                                  placeholder="gsk_...", label_visibility="collapsed")
        if key_input:
            os.environ["GROQ_API_KEY"] = key_input

        has_key = bool(key_input or os.getenv("GROQ_API_KEY"))
        badge_class = "badge-online" if has_key else "badge-offline"
        badge_text = "🟢 API Connected" if has_key else "🔴 No API Key"
        st.markdown(f'<span class="badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)

        db_stats_placeholder = st.empty()
        example_tips_placeholder = st.empty()

        st.markdown('<div class="sidebar-section">🕘 Query History</div>', unsafe_allow_html=True)
        user_turns = [t for t in st.session_state["history"] if t["role"] == "user"]
        if user_turns:
            for i, t in enumerate(reversed(user_turns[-8:])):
                q_short = t["content"][:45] + "…" if len(t["content"]) > 45 else t["content"]
                st.markdown(f'<div class="hist-item">#{len(user_turns)-i} {q_short}</div>',
                            unsafe_allow_html=True)
        else:
            st.caption("No queries yet. Ask something above! ☝️")

        st.markdown('<div class="sidebar-section">📊 Session Stats</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">SQL Runs</div>
                <div class="metric-value">{st.session_state['query_count']}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Chat Turns</div>
                <div class="metric-value">{len(st.session_state['history'])}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
        if st.button("🗑️  Clear Chat", use_container_width=True):
            st.session_state["history"] = []
            st.session_state["query_count"] = 0
            st.rerun()

        st.markdown("""
        <div style='text-align:center; margin-top:20px; color:#30363d; font-size:0.72rem;'>
            Powered by LLaMA 3.3 · Groq · Streamlit<br>
            <span style='color:#2196f3;'>EMAN SQL AGENT</span> © 2026
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  MAIN PANEL HEADER
    # ══════════════════════════════════════════════════════
    col_title, col_model = st.columns([3, 1])
    with col_title:
        st.markdown("""
        <div class="eman-title">⚡EMAN SQL AGENT</div>
        <div class="eman-subtitle">
            Your intelligent database companion — turning plain questions into powerful SQL
        </div>
        <div class="eman-features">
            <span class="feat-pill">📂 Upload any <b>.db</b> file</span>
            <span class="feat-pill">⚙️ See exact SQL query</span>
            <span class="feat-pill">🧠 Watch agent think live</span>
            <span class="feat-pill">💡 Smart follow-ups</span>
        </div>
        """, unsafe_allow_html=True)
    with col_model:
        st.markdown("""
        <div style='text-align:right; padding-top:10px;'>
            <span class="badge badge-online" style="font-size:0.7rem;">
                🤖 llama-3.3-70b-versatile
            </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  DATABASE LOAD
    # ══════════════════════════════════════════════════════
    db_path = get_db_path(uploaded)
    if not db_path:
        st.warning("⚠️ No database found. Upload a `.db` file from sidebar or add `student.db` here.")
        return

    try:
        conn = connect_db(db_path)
    except Exception as e:
        st.error(f"❌ Could not open database: {e}")
        return

    schema = get_schema(conn)
    if not schema:
        st.error("❌ No user tables found in this database.")
        return

    db_stats = get_db_stats(conn, schema)
    total_rows = sum(v for v in db_stats.values() if isinstance(v, int))
    with db_stats_placeholder.container():
        st.markdown('<div class="sidebar-section">🗄️ Database Info</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tables</div>
                <div class="metric-value">{len(schema)}</div>
            </div>""", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{total_rows}</div>
            </div>""", unsafe_allow_html=True)
        for tbl, cnt in db_stats.items():
            st.markdown(f'<div class="hist-item">📋 <b>{tbl}</b> — {cnt} rows</div>',
                        unsafe_allow_html=True)

    with st.expander("📐 View Database Schema", expanded=False):
        schema_txt = schema_to_text(schema)
        st.markdown(f'<div class="schema-box">{schema_txt}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  LLM
    # ══════════════════════════════════════════════════════
    try:
        llm = get_llm(os.getenv("GROQ_API_KEY"))
    except Exception as e:
        st.error(f"❌ {e}")
        return

    schema_text = schema_to_text(schema)
    current_schema_key = schema_text.strip()

    if st.session_state["last_schema_key"] != current_schema_key:
        with st.spinner("💡 Generating smart questions for your database..."):
            st.session_state["example_tips"] = generate_example_questions(llm, schema_text)
            st.session_state["last_schema_key"] = current_schema_key

    with example_tips_placeholder.container():
        st.markdown('<div class="sidebar-section">💡 Example Questions</div>', unsafe_allow_html=True)
        for tip in st.session_state["example_tips"]:
            if st.button(f"▸  {tip}", key=f"tip_{tip}", use_container_width=True):
                st.session_state["pending_question"] = tip

    # ══════════════════════════════════════════════════════
    #  CHAT HISTORY DISPLAY
    # ══════════════════════════════════════════════════════
    for turn in st.session_state["history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])


    # ══════════════════════════════════════════════════════
    #  HANDLE INPUT
    # ══════════════════════════════════════════════════════
    user_q = st.chat_input("💬  Ask about your data in English or Roman Urdu...")

    # pending_question (sidebar tips ya followup) ko user_q bana do
    if not user_q and st.session_state.get("pending_question"):
        user_q = st.session_state["pending_question"]
        st.session_state["pending_question"] = None

    if user_q:
        # Followup buttons hide karo jab naya question aa jaye
        st.session_state["last_followups"] = []

        with st.chat_message("user"):
            st.markdown(user_q)
        st.session_state["history"].append({"role": "user", "content": user_q})
        st.session_state["query_count"] += 1

        with st.chat_message("assistant"):
            st.markdown("⚡ **Eman SQL Agent is analyzing your question…**")

            plan      = ask_llm_for_sql(llm, user_q, schema_text)
            sql       = plan.get("sql", "")
            thinking  = plan.get("thinking", "")
            followups = plan.get("followups", [])[:3]

            if thinking:
                st.markdown(
                    f'<div class="think-box">🧠 <b>Agent Thinking:</b> {thinking}</div>',
                    unsafe_allow_html=True)

            if sql:
                st.markdown("**⚙️ Generated SQL Query:**")
                st.code(sql, language="sql")
            else:
                st.warning("No SQL was generated.")

            result = run_sql(conn, sql) if sql else "No SQL to run."

            if isinstance(result, pd.DataFrame):
                if result.empty:
                    st.info("✅ Query ran successfully but returned **0 rows**.")
                else:
                    row_count = len(result)
                    st.markdown(
                        f'<span class="badge badge-online">✓ {row_count} row(s) returned</span>',
                        unsafe_allow_html=True)
                    st.dataframe(result, use_container_width=True)
                    csv = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="eman_query_result.csv",
                        mime="text/csv",
                    )
            else:
                if "Error" in str(result) or "Blocked" in str(result):
                    st.error(result)
                else:
                    st.write(result)

            final_answer = build_final_answer(llm, user_q, sql, result)
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown(final_answer)

        if followups:
                st.session_state["last_followups"] = followups

        st.session_state["history"].append({"role": "assistant", "content": final_answer})

    if st.session_state.get("last_followups"):
        st.markdown("**🔍 You might also want to know:**")
        cols = st.columns(len(st.session_state["last_followups"]))
        for i, fq in enumerate(st.session_state["last_followups"]):
            if cols[i].button(fq, key=f"fup_persistent_{i}"):
                st.session_state["last_followups"] = []
                st.session_state["pending_question"] = fq
                st.rerun()


if __name__ == "__main__":
    main()