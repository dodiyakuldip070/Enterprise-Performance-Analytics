import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import hashlib
from datetime import datetime
import plotly.express as px

# ----------------------
# Configuration
# ----------------------
USERS_DIR = "users"

MAX_SAMPLE_ROWS = 150000  # threshold for downsampling

os.makedirs(USERS_DIR, exist_ok=True)

# ----------------------
# Utility functions (auth)
# ----------------------

def hash_password(password: str, salt: str = None):
    """Return (salt, sha256(salt+password)). If salt is None generate one."""
    if salt is None:
        salt = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
    hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return salt, hashed


def user_file(username: str) -> str:
    return os.path.join(USERS_DIR, f"{username}.json")


def create_user(username: str, password: str, full_name: str = "") -> bool:
    path = user_file(username)
    if os.path.exists(path):
        return False
    salt, hashed = hash_password(password)
    data = {
        "username": username,
        "full_name": full_name,
        "salt": salt,
        "password": hashed,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return True


def verify_user(username: str, password: str) -> bool:
    path = user_file(username)
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        data = json.load(f)
    salt = data.get("salt")
    _, hashed = hash_password(password, salt)
    return hashed == data.get("password")


def get_display_name(username: str) -> str:
    path = user_file(username)
    if not os.path.exists(path):
        return username
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("full_name") or data.get("username")

# ----------------------
# Data helpers
# ----------------------

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    # file can be a path or an uploaded file-like object
    if isinstance(file, str):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)
    return df


def numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_cols(df: pd.DataFrame):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()

# ----------------------
# Session helpers
# ----------------------

def init_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "filtered_df" not in st.session_state:
        st.session_state.filtered_df = None


def do_logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.df = None
    st.session_state.filtered_df = None
    st.rerun()

# ----------------------
# Login / Signup UI (right-side compact, Facebook-like)
# ----------------------

def _inject_auth_css():
    css = """
    <style>
    /* Right-panel styling */
    .right-panel {
      background: #ffffff;
      padding: 22px;
      border-radius: 12px;
      box-shadow: 0 6px 20px rgba(0,0,0,0.08);
      max-width: 420px;
      margin-left: auto;
    }
    .brand-panel {
      background: linear-gradient(180deg,#1877F2 0%, #145dbf 100%);
      color: white;
      padding: 34px;
      border-radius: 12px;
      height: 100%;
    }
    input[type="text"], input[type="password"], textarea {
      font-weight: 700 !important;
      padding: 12px !important;
      border-radius: 8px !important;
    }
    .stButton>button {
      font-weight: 800 !important;
      padding: 10px 14px !important;
      border-radius: 8px !important;
      background-color: #1877F2 !important;
      color: white !important;
    }
    .small-hint {
      color: rgba(0,0,0,0.6);
      font-size: 13px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def page_login():
    _inject_auth_css()
    # Layout: left brand, right compact form
    col_brand, col_form = st.columns([2, 1])

    with col_brand:
        st.markdown(
            "<div class='brand-panel'>"
            "<h1 style='margin:0; font-size:40px; font-weight:800;'>Enterprise Performance Analytics</h1>"
            "<p style='margin-top:8px; font-size:16px; opacity:0.95;'>Data-driven insights for strategic growth</p>"
            "</div>", unsafe_allow_html=True
        )

    with col_form:
        st.markdown("<div class='right-panel'>", unsafe_allow_html=True)
        with st.form("login_form"):
            st.markdown("<h3 style='text-align:center; margin-bottom:6px;'>Login</h3>", unsafe_allow_html=True)
            username = st.text_input("", placeholder="Username", key="login_user")
            password = st.text_input("", placeholder="Password", type="password", key="login_pass")
            st.write("<div class='small-hint' style='text-align:center'>Log in to access your dashboard</div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Log In")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            if verify_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in")
                st.rerun()
            else:
                st.error("Invalid username or password")


def page_signup():
    _inject_auth_css()
    col_brand, col_form = st.columns([2, 1])

    with col_brand:
        st.markdown(
            "<div class='brand-panel'>"
            "<h1 style='margin:0; font-size:40px; font-weight:800;'>Enterprise Performance Analytics</h1>"
            "<p style='margin-top:8px; font-size:16px; opacity:0.95;'>Join to unlock analytics</p>"
            "</div>", unsafe_allow_html=True
        )

    with col_form:
        st.markdown("<div class='right-panel'>", unsafe_allow_html=True)
        with st.form("signup_form"):
            st.markdown("<h3 style='text-align:center; margin-bottom:6px;'>Sign Up</h3>", unsafe_allow_html=True)
            username = st.text_input("", placeholder="Choose a username", key="signup_user")
            full_name = st.text_input("", placeholder="Full name (optional)", key="signup_full")
            password = st.text_input("", placeholder="Create a password", type="password", key="signup_pass")
            password2 = st.text_input("", placeholder="Repeat password", type="password", key="signup_pass2")
            st.write("<div class='small-hint' style='text-align:center'>We store only a hashed password per user on-disk for this demo.</div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Create account")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            if not username or not password:
                st.error("Username and password required")
            elif password != password2:
                st.error("Passwords do not match")
            else:
                ok = create_user(username, password, full_name)
                if ok:
                    st.success("Account created. Please login.")
                else:
                    st.error("Username already exists")

# ----------------------
# Other app pages (unchanged behavior)
# ----------------------



def page_dashboard():
    # Top navbar
    cols = st.columns([1, 6, 1])
    cols[1].markdown("# Enterprise Performance Analytics")
    right = cols[2]
    right.write("")
    if st.session_state.logged_in:
        right.markdown(f"**{get_display_name(st.session_state.username)}**")
        if right.button("Logout"):
            do_logout()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    nav = st.sidebar.radio("Go to", ["Data Preview", "Data Search & Filter", "Data Visualization", "Data Insights"])

    # Data handling area: uploader and sample
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Drag & drop or choose CSV file", type=["csv"], key="uploader")
   
   
    if uploaded_file is not None:
        try:
            df = load_csv(uploaded_file)
            st.session_state.df = df
            st.success("Uploaded dataset loaded into memory (not saved on server).")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

    df = st.session_state.get("df")

    if nav == "Data Preview":
        st.header("Data Preview")
        if df is None:
            st.info("No dataset loaded. Use the sidebar to upload or load the sample dataset.")
            return
        # Show quick metrics
        st.subheader("Quick summary metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]}")
        missing = df.isna().sum().sum()
        c3.metric("Missing values", f"{missing:,}")
        mem = df.memory_usage(deep=True).sum()
        c4.metric("Memory (bytes)", f"{mem:,}")

        # Show column types and head
        st.subheader("Columns & types")
        types = pd.DataFrame({"column": df.columns, "dtype": [str(x) for x in df.dtypes]})
        st.dataframe(types)

        st.subheader("Preview")
        st.dataframe(df)


        st.subheader("Column statistics")
        st.write(df.describe(include="all").T)

    elif nav == "Data Search & Filter":
        st.header("Data Search & Filter")
        if df is None:
            st.info("Load dataset in the sidebar first.")
            return
        st.markdown("Use the controls below to filter the dataset. The filtered dataset is used by visualizations.")
        # Basic search across text columns
        text_cols = categorical_cols(df)
        numeric = numeric_cols(df)

        # Full-text search
        q = st.text_input("Full-text search (matches any selected text columns)")
        cols_for_search = st.multiselect("Columns to search", text_cols, default=text_cols[:3]) if text_cols else []

        filtered = df
        if q and cols_for_search:
            mask = pd.Series(False, index=df.index)
            for col in cols_for_search:
                mask = mask | df[col].astype(str).str.contains(q, case=False, na=False)
            filtered = df[mask]

        # Numeric range filters
        st.subheader("Numeric filters")
        numeric_filters = {}
        for col in numeric:
            mn = float(df[col].min())
            mx = float(df[col].max())
            lo, hi = st.slider(f"{col}", min_value=mn, max_value=mx, value=(mn, mx), key=f"num_{col}")
            numeric_filters[col] = (lo, hi)
            filtered = filtered[(filtered[col] >= lo) & (filtered[col] <= hi)]

        st.subheader("Filtered dataset sample")
        st.write(f"Rows after filter: {len(filtered):,}")
        st.dataframe(filtered.head(200))
        st.session_state.filtered_df = filtered

    elif nav == "Data Visualization":
        st.header("Data Visualization")
        df_vis = st.session_state.get("filtered_df") if st.session_state.get("filtered_df") is not None else st.session_state.get("df")
        if df_vis is None:
            st.info("Load dataset first (sidebar) or apply filters in 'Data Search & Filter'.")
            return

        st.subheader("Performance settings")
        max_points = st.number_input("Max points to plot (will sample if dataset larger)", min_value=1000, max_value=1000000, value=200000, step=1000)

        # downsample if needed
        plot_df = df_vis
        if len(plot_df) > max_points:
            plot_df = plot_df.sample(n=max_points, random_state=42)
            st.warning(f"Plotting sample of {max_points} rows for performance (dataset has {len(df_vis):,} rows).")

        chart_type = st.selectbox("Chart type", ["Scatter", "Line", "Bar", "Bubble", "Histogram", "Box", "Pie"]) 

        if chart_type == "Scatter":
            st.write("Scatter plot — choose X and Y")
            num_cols = numeric_cols(plot_df)
            if len(num_cols) < 2:
                st.warning("Need at least two numeric columns for scatter plot.")
            else:
                x = st.selectbox("X", num_cols, index=0)
                y = st.selectbox("Y", num_cols, index=1)
                color = st.selectbox("Color (optional)", [None] + categorical_cols(plot_df))
                size = st.selectbox("Size (optional)", [None] + num_cols)
                fig = px.scatter(plot_df, x=x, y=y, color=color if color else None, size=size if size else None, title=f"Scatter: {y} vs {x}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line":
            st.write("Line chart — choose a numeric column (aggregated by index or a time column)")
            num_cols = numeric_cols(plot_df)
            if not num_cols:
                st.warning("No numeric columns available.")
            else:
                y = st.selectbox("Y", num_cols)
                x_candidates = [c for c in plot_df.columns if np.issubdtype(plot_df[c].dtype, np.datetime64)]
                x = None
                if x_candidates:
                    x = st.selectbox("X (time)", [None] + x_candidates)
                if x:
                    fig = px.line(plot_df.sort_values(x), x=x, y=y, title=f"Line: {y} over {x}")
                else:
                    # aggregate by index
                    agg = plot_df[y].reset_index().groupby("index")[y].mean()
                    fig = px.line(agg, y=y, title=f"Line (index aggregated): {y}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar":
            st.write("Bar chart — choose categorical and numeric")
            cat_cols = categorical_cols(plot_df)
            num_cols = numeric_cols(plot_df)
            if not cat_cols or not num_cols:
                st.warning("Need a categorical and numeric column for bar chart.")
            else:
                cat = st.selectbox("Category", cat_cols)
                agg = st.selectbox("Aggregation", ["sum", "mean", "median", "count"]) 
                val = st.selectbox("Value (numeric)", num_cols)
                df_bar = plot_df.groupby(cat)[val].agg(agg).reset_index()
                fig = px.bar(df_bar, x=cat, y=val, title=f"Bar: {agg} of {val} by {cat}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bubble":
            st.write("Bubble chart — X, Y, size and (optional) color")
            num_cols = numeric_cols(plot_df)
            if len(num_cols) < 3:
                st.warning("Need at least three numeric columns for a bubble chart (X, Y, Size).")
            else:
                x = st.selectbox("X", num_cols, index=0)
                y = st.selectbox("Y", num_cols, index=1)
                size = st.selectbox("Size", num_cols, index=2)
                color = st.selectbox("Color (optional)", [None] + categorical_cols(plot_df))
                fig = px.scatter(plot_df, x=x, y=y, size=size, color=color if color else None, title=f"Bubble: {y} vs {x}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Histogram":
            st.write("Histogram — choose numeric column")
            num_cols = numeric_cols(plot_df)
            if not num_cols:
                st.warning("No numeric columns available.")
            else:
                col = st.selectbox("Column", num_cols)
                bins = st.slider("Bins", 5, 500, 50)
                fig = px.histogram(plot_df, x=col, nbins=bins, title=f"Histogram of {col}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box":
            st.write("Box plot — choose numeric column (and optional category)")
            num_cols = numeric_cols(plot_df)
            if not num_cols:
                st.warning("No numeric columns available.")
            else:
                col = st.selectbox("Numeric", num_cols)
                cat = st.selectbox("Category (optional)", [None] + categorical_cols(plot_df))
                fig = px.box(plot_df, x=cat if cat else None, y=col, title=f"Box: {col} by {cat if cat else 'all'}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pie":
            st.write("Pie chart — choose categorical column")
            cat_cols = categorical_cols(plot_df)
            if not cat_cols:
                st.warning("No categorical columns available.")
            else:
                col = st.selectbox("Category", cat_cols)
                df_pie = plot_df[col].value_counts().reset_index()
                df_pie.columns = [col, "count"]
                fig = px.pie(df_pie, names=col, values="count", title=f"Pie: {col}")
                st.plotly_chart(fig, use_container_width=True)

    elif nav == "Data Insights":
        st.header("Data Insights")
        df_ins = st.session_state.get("filtered_df") if st.session_state.get("filtered_df") is not None else st.session_state.get("df")
        if df_ins is None:
            st.info("Load dataset first.")
            return
        st.subheader("Top correlations (numeric)")
        num = numeric_cols(df_ins)
        if len(num) >= 2:
            corr = df_ins[num].corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
            corr = corr[corr < 1].dropna().head(20)
            corr_df = corr.reset_index()
            corr_df.columns = ["A", "B", "abs_corr"]
            st.dataframe(corr_df)
        else:
            st.info("Not enough numeric columns to compute correlations.")

        st.subheader("Missing data report")
        miss = df_ins.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if len(miss) == 0:
            st.success("No missing values found.")
        else:
            st.dataframe(miss)

        st.subheader("Basic trend detection (numeric columns)")
        for c in numeric_cols(df_ins)[:6]:
            st.write(f"Column: {c}")
            s = df_ins[c].dropna()
            if len(s) > 2:
                # simple slope via linear fit on sample
                x = np.arange(len(s))
                lr = np.polyfit(x, s.values, 1)
                slope = lr[0]
                st.write(f"Slope (simple): {slope:.6g}")

    elif nav == "About":
        st.header("About this demo")
        st.markdown(
            "This Streamlit demo implements:\n- File-based signup/login (per-user JSON file)\n- Sidebar navigation and top navbar with logged-in name\n- Data preview, search & filter, visualizations and insights\n\nThe sample dataset path is /mnt/data/Walmart.csv for demonstration only. Uploaded files are loaded into memory and not stored on the server. For production you should replace the simple file-based auth with a secure backend and hashed password storage (e.g. bcrypt) and secure session management."
        )

# ----------------------
# App entry point
# ----------------------

def main():
    st.set_page_config(page_title="Enterprise Performance Analytics", layout="wide")
    init_session_state()

    if not st.session_state.logged_in:
        tabs = st.tabs(["Login", "Sign up"])
        with tabs[0]:
            page_login()
        with tabs[1]:
            page_signup()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()