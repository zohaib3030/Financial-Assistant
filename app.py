import os
import re
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Assistant",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 0;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 25px;
        }

        .metric-card {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #ddd;
            background: #fafafa;
        }

        .category-card {
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
        }

        .stButton button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TRAINING DATA
# ============================================================

TRAINING_DATA = [
    # Food
    ("breakfast at restaurant", "Food"),
    ("lunch at restaurant", "Food"),
    ("dinner", "Food"),
    ("pizza", "Food"),
    ("burger", "Food"),
    ("mcdonalds", "Food"),
    ("kfc", "Food"),
    ("biryani", "Food"),
    ("groceries", "Food"),
    ("milk", "Food"),
    ("bread", "Food"),
    ("vegetables", "Food"),
    ("fruits", "Food"),
    ("snacks", "Food"),
    ("coffee", "Food"),
    ("tea", "Food"),
    ("restaurant bill", "Food"),
    ("food delivery", "Food"),
    ("ordered food", "Food"),
    ("eating out", "Food"),

    # Transportation
    ("uber ride", "Transportation"),
    ("careem ride", "Transportation"),
    ("taxi", "Transportation"),
    ("bus fare", "Transportation"),
    ("train ticket", "Transportation"),
    ("fuel", "Transportation"),
    ("petrol", "Transportation"),
    ("diesel", "Transportation"),
    ("car fuel", "Transportation"),
    ("bike fuel", "Transportation"),
    ("rickshaw", "Transportation"),
    ("transport fare", "Transportation"),
    ("parking fee", "Transportation"),
    ("car repair", "Transportation"),
    ("bike repair", "Transportation"),

    # Shopping
    ("new shirt", "Shopping"),
    ("new shoes", "Shopping"),
    ("clothes", "Shopping"),
    ("shopping mall", "Shopping"),
    ("jeans", "Shopping"),
    ("jacket", "Shopping"),
    ("watch", "Shopping"),
    ("bag", "Shopping"),
    ("online shopping", "Shopping"),
    ("amazon purchase", "Shopping"),
    ("daraz order", "Shopping"),
    ("electronics", "Shopping"),
    ("headphones", "Shopping"),
    ("mobile accessories", "Shopping"),

    # Bills
    ("electricity bill", "Bills"),
    ("gas bill", "Bills"),
    ("water bill", "Bills"),
    ("internet bill", "Bills"),
    ("wifi bill", "Bills"),
    ("phone bill", "Bills"),
    ("mobile bill", "Bills"),
    ("utility bill", "Bills"),
    ("electricity payment", "Bills"),
    ("internet subscription", "Bills"),

    # Entertainment
    ("movie ticket", "Entertainment"),
    ("cinema", "Entertainment"),
    ("netflix", "Entertainment"),
    ("spotify", "Entertainment"),
    ("gaming", "Entertainment"),
    ("game purchase", "Entertainment"),
    ("concert", "Entertainment"),
    ("party", "Entertainment"),
    ("entertainment", "Entertainment"),
    ("youtube premium", "Entertainment"),

    # Education
    ("college fee", "Education"),
    ("university fee", "Education"),
    ("school fee", "Education"),
    ("books", "Education"),
    ("textbooks", "Education"),
    ("course fee", "Education"),
    ("online course", "Education"),
    ("udemy course", "Education"),
    ("programming course", "Education"),
    ("stationery", "Education"),
    ("notebooks", "Education"),
    ("exam fee", "Education"),

    # Health
    ("doctor appointment", "Health"),
    ("doctor fee", "Health"),
    ("medicine", "Health"),
    ("medicines", "Health"),
    ("hospital bill", "Health"),
    ("pharmacy", "Health"),
    ("medical test", "Health"),
    ("dental treatment", "Health"),
    ("gym membership", "Health"),
    ("health checkup", "Health"),

    # Housing
    ("house rent", "Housing"),
    ("monthly rent", "Housing"),
    ("room rent", "Housing"),
    ("apartment rent", "Housing"),
    ("hostel fee", "Housing"),
    ("home repair", "Housing"),
    ("furniture", "Housing"),

    # Personal Care
    ("haircut", "Personal Care"),
    ("barber", "Personal Care"),
    ("shampoo", "Personal Care"),
    ("soap", "Personal Care"),
    ("perfume", "Personal Care"),
    ("skin care", "Personal Care"),
    ("cosmetics", "Personal Care"),
    ("salon", "Personal Care"),

    # Other
    ("gift", "Other"),
    ("donation", "Other"),
    ("miscellaneous", "Other"),
    ("other expense", "Other"),
]


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model():

    texts = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000
            )
        )
    ])

    model.fit(texts, labels)

    return model


model = train_model()


# ============================================================
# EXPENSE PARSING
# ============================================================

def extract_amount(text):
    """
    Extract the first numeric amount from an expense description.
    Examples:
        "Lunch 500" -> 500
        "Uber - Rs 800" -> 800
        "Netflix 1500" -> 1500
    """

    matches = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", text)

    if not matches:
        return 0.0

    amount = matches[-1].replace(",", "")

    try:
        return float(amount)
    except ValueError:
        return 0.0


def clean_expense_text(text):
    """
    Remove amounts from text before sending it to the ML model.
    """

    cleaned = re.sub(
        r"rs\.?\s*\d+(?:,\d+)*(?:\.\d+)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\d+(?:,\d+)*(?:\.\d+)?",
        "",
        cleaned
    )

    return cleaned.strip()


# ============================================================
# CATEGORY PREDICTION
# ============================================================

def predict_category(expense_text):

    cleaned_text = clean_expense_text(expense_text)

    if not cleaned_text:
        cleaned_text = expense_text

    prediction = model.predict([cleaned_text])[0]

    probabilities = model.predict_proba([cleaned_text])[0]

    confidence = max(probabilities) * 100

    return prediction, confidence


# ============================================================
# FALLBACK FINANCIAL ANALYSIS
# ============================================================

def fallback_strategy(expenses_df):

    total = expenses_df["Amount"].sum()

    category_totals = (
        expenses_df
        .groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    largest_category = category_totals.index[0]
    largest_amount = category_totals.iloc[0]

    percentage = (
        largest_amount / total * 100
        if total > 0
        else 0
    )

    strategies = []

    strategies.append(
        f"Your largest spending category is **{largest_category}**, "
        f"accounting for approximately **{percentage:.1f}%** of your total spending."
    )

    if "Food" in category_totals:
        food_percentage = category_totals["Food"] / total * 100

        if food_percentage > 20:
            strategies.append(
                "Your food spending is relatively high. "
                "Try reducing restaurant and food-delivery orders and prepare more meals at home."
            )

    if "Entertainment" in category_totals:
        entertainment_percentage = (
            category_totals["Entertainment"] / total * 100
        )

        if entertainment_percentage > 10:
            strategies.append(
                "Review your entertainment subscriptions and cancel services you rarely use."
            )

    if "Shopping" in category_totals:
        shopping_percentage = (
            category_totals["Shopping"] / total * 100
        )

        if shopping_percentage > 15:
            strategies.append(
                "Set a monthly shopping limit and use a 24-hour waiting rule "
                "before making non-essential purchases."
            )

    strategies.append(
        "Create a monthly budget and assign a spending limit to each major category."
    )

    strategies.append(
        "Try to save at least 10–20% of your income before spending on discretionary items."
    )

    return "\n\n".join(
        [f"- {strategy}" for strategy in strategies]
    )


# ============================================================
# GROQ LLM ANALYSIS
# ============================================================

def generate_llm_strategy(expenses_df, api_key):

    try:

        from groq import Groq

        client = Groq(api_key=api_key)

        category_summary = (
            expenses_df
            .groupby("Category")["Amount"]
            .sum()
            .sort_values(ascending=False)
        )

        expense_text = "\n".join(
            [
                f"- {row['Description']} | "
                f"{row['Amount']:.0f} | "
                f"{row['Category']}"
                for _, row in expenses_df.iterrows()
            ]
        )

        category_text = "\n".join(
            [
                f"- {category}: {amount:.0f}"
                for category, amount in category_summary.items()
            ]
        )

        total = expenses_df["Amount"].sum()

        prompt = f"""
You are a professional personal finance assistant.

Analyze the user's daily expenses and provide practical,
realistic money-saving strategies.

Total spending:
{total:.0f}

Category breakdown:
{category_text}

Individual expenses:
{expense_text}

Your response must contain:

1. Spending Overview
2. Biggest Spending Problems
3. 5 Personalized Money-Saving Strategies
4. Suggested Budget Adjustments
5. A simple 30-day saving challenge

Do not provide investment, tax, loan, or legally regulated financial advice.

Keep the advice practical for an ordinary person.
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful personal finance assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=1500
        )

        return response.choices[0].message.content

    except Exception as e:

        st.warning(
            "LLM analysis could not be generated. "
            "Showing local financial analysis instead."
        )

        return fallback_strategy(expenses_df)


# ============================================================
# SESSION STATE
# ============================================================

if "expenses" not in st.session_state:
    st.session_state.expenses = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    currency = st.selectbox(
        "Currency",
        ["PKR", "USD", "EUR", "GBP", "INR"]
    )


    

    groq_key = st.secrets.get("GROQ_API_KEY", "")

    

    st.divider()

    if st.button("🗑️ Clear All Expenses"):


        st.session_state.expenses = []

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="title">💰 Financial Assistant</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    "Track your spending • Understand your habits • Save smarter"
    "</p>",
    unsafe_allow_html=True
)


# ============================================================
# EXPENSE INPUT
# ============================================================

st.subheader("📝 Add Daily Expenses")

st.info(
    "Enter one expense per line. Include the amount at the end."
)

expense_input = st.text_area(
    "Expenses",
    placeholder="""Examples:

Breakfast at restaurant 500
Uber ride 800
Electricity bill 4500
Netflix subscription 1500
New shoes 6500
Python course 3000""",
    height=220
)


if st.button("➕ Analyze Expenses", type="primary"):

    if not expense_input.strip():

        st.error("Please enter at least one expense.")

    else:

        lines = [
            line.strip()
            for line in expense_input.split("\n")
            if line.strip()
        ]

        new_expenses = []

        for line in lines:

            amount = extract_amount(line)

            if amount <= 0:
                continue

            category, confidence = predict_category(line)

            new_expenses.append(
                {
                    "Description": line,
                    "Amount": amount,
                    "Category": category,
                    "Confidence": confidence
                }
            )

        if not new_expenses:

            st.error(
                "Could not detect any valid amounts. "
                "Please enter expenses such as 'Lunch 500'."
            )

        else:

            st.session_state.expenses.extend(new_expenses)

            st.success(
                f"Successfully analyzed {len(new_expenses)} expense(s)."
            )


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.expenses:

    df = pd.DataFrame(st.session_state.expenses)

    total_spending = df["Amount"].sum()

    number_expenses = len(df)

    average_expense = df["Amount"].mean()

    highest_expense = df["Amount"].max()


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.divider()

    st.subheader("📊 Financial Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Spending",
            f"{currency} {total_spending:,.0f}"
        )

    with col2:
        st.metric(
            "Number of Expenses",
            number_expenses
        )

    with col3:
        st.metric(
            "Average Expense",
            f"{currency} {average_expense:,.0f}"
        )

    with col4:
        st.metric(
            "Highest Expense",
            f"{currency} {highest_expense:,.0f}"
        )


    # --------------------------------------------------------
    # CATEGORY ANALYSIS
    # --------------------------------------------------------

    st.divider()

    left, right = st.columns(2)

    category_summary = (
        df
        .groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    with left:

        st.subheader("📂 Spending by Category")

        chart_data = category_summary.to_frame(
            name="Amount"
        )

        st.bar_chart(chart_data)


    with right:

        st.subheader("🥧 Category Distribution")

        percentage_data = (
            category_summary / total_spending * 100
        ).round(2)

        percentage_df = pd.DataFrame(
            {
                "Category": percentage_data.index,
                "Percentage": percentage_data.values
            }
        )

        st.dataframe(
            percentage_df,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # EXPENSE TABLE
    # --------------------------------------------------------

    st.divider()

    st.subheader("📋 Expense History")

    display_df = df.copy()

    display_df["Amount"] = display_df["Amount"].apply(
        lambda x: f"{currency} {x:,.0f}"
    )

    display_df["Confidence"] = display_df["Confidence"].apply(
        lambda x: f"{x:.1f}%"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # AI FINANCIAL ADVISOR
    # --------------------------------------------------------

    st.divider()

    st.subheader("🧠 AI Financial Advisor")

    st.write(
        "Let AI analyze your spending pattern and suggest "
        "personalized ways to save money."
    )

    if st.button("✨ Generate Saving Strategy"):

        with st.spinner(
            "Analyzing your financial habits..."
        ):

            if groq_key.strip():

                strategy = generate_llm_strategy(
                    df,
                    groq_key.strip()
                )

            else:

                strategy = fallback_strategy(df)

            st.session_state.strategy = strategy


    if "strategy" in st.session_state:

        st.markdown("### 💡 Your Personalized Strategy")

        st.markdown(
            st.session_state.strategy
        )


else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.divider()

    st.markdown(
        """
        ### 👋 Welcome to your Financial Assistant!

        Start by entering your daily expenses above.

        **Example:**

        `Lunch 500`

        `Uber 700`

        `Electricity bill 4500`

        `Netflix 1500`

        `New shoes 6000`

        The application will automatically:

        **Expense → ML Categorization → Spending Analysis → AI Strategy → Savings**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "💰 Financial Assistant | Built with Python, Streamlit, "
    "Machine Learning & Generative AI"
)
