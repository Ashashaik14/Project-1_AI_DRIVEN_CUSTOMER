from flask import Flask, render_template, request, redirect, jsonify, send_file
import pandas as pd
import numpy as np
import io

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("dataset/customers.csv")


# =====================================================
# SEGMENTATION
# =====================================================
def create_segments(data):
    data = data.copy()

    features = data[["annual_income", "spending_score"]]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data["cluster"] = kmeans.fit_predict(features)

    segment_map = {0: "Budget", 1: "Regular", 2: "Premium"}
    data["segment"] = data["cluster"].map(segment_map)

    return data


# =====================================================
# CHURN
# =====================================================
def create_churn(data):
    data = data.copy()

    conditions = [
        data["last_purchase_days"] > 20,
        data["last_purchase_days"] > 10
    ]

    choices = ["High Risk", "Medium Risk"]

    data["churn"] = np.select(conditions, choices, default="Low Risk")
    return data


# apply transformations
df = create_segments(df)
df = create_churn(df)


# =====================================================
# ML MODEL
# =====================================================
X = df[["age", "annual_income", "spending_score", "purchase_amount"]]

y = np.where(df["segment"] == "Premium", 1, 0)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

prediction_history = []


# =====================================================
# HOME
# =====================================================
@app.route("/")
def home():
    return redirect("/login")


# =====================================================
# LOGIN (FIXED PASSWORD)
# =====================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # FIXED LOGIN (admin/admin123)
        if username == "admin" and password == "admin123":
            return redirect("/dashboard")

        error = "Invalid Credentials"

    return render_template("login.html", error=error)


# =====================================================
# DASHBOARD
# =====================================================
@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html",
        total=len(df),
        avg_income=round(df["annual_income"].mean(), 2),
        avg_spending=round(df["spending_score"].mean(), 2),
        total_revenue=int(df["purchase_amount"].sum()),

        gender_counts=df["gender"].value_counts().to_dict(),
        cluster_counts=df["segment"].value_counts().to_dict(),
        churn_counts=df["churn"].value_counts().to_dict(),

        age_counts=pd.cut(df["age"],
                          bins=[18, 25, 35, 45, 60],
                          labels=["18-25", "26-35", "36-45", "46+"]
                          ).value_counts().sort_index().to_dict(),

        income_counts=pd.cut(df["annual_income"],
                             bins=[0, 40000, 60000, 80000, 100000],
                             labels=["<40K", "40K-60K", "60K-80K", "80K+"]
                             ).value_counts().sort_index().to_dict(),

        spending_counts=pd.cut(df["spending_score"],
                               bins=[0, 40, 60, 80, 100],
                               labels=["Low", "Medium", "High", "Premium"]
                               ).value_counts().sort_index().to_dict(),

        table_data=df.to_dict(orient="records")
    )


# =====================================================
# CUSTOMERS (FIXED - ADD ALL COUNTS)
# =====================================================
@app.route("/customers")
def customers():

    return render_template(
        "customers.html",
        customers=df.to_dict(orient="records"),

        gender_counts=df["gender"].value_counts().to_dict(),

        age_counts=pd.cut(df["age"],
                          bins=[18, 25, 35, 45, 60],
                          labels=["18-25", "26-35", "36-45", "46+"]
                          ).value_counts().sort_index().to_dict(),

        income_counts=pd.cut(df["annual_income"],
                             bins=[0, 40000, 60000, 80000, 100000],
                             labels=["<40K", "40K-60K", "60K-80K", "80K+"]
                             ).value_counts().sort_index().to_dict(),

        spending_counts=pd.cut(df["spending_score"],
                               bins=[0, 40, 60, 80, 100],
                               labels=["Low", "Medium", "High", "Premium"]
                               ).value_counts().sort_index().to_dict(),

        purchase_counts=pd.cut(df["purchase_amount"],
                               bins=[0, 300, 600, 900, 1500],
                               labels=["Low", "Medium", "High", "Premium"]
                               ).value_counts().sort_index().to_dict(),

        avg_purchase=round(df["purchase_amount"].mean(), 2)
    )


# =====================================================
# SEGMENTATION (FIXED)
# =====================================================
@app.route("/segmentation")
def segmentation():

    return render_template(
        "segmentation.html",
        data=df.to_dict(orient="records"),

        segment_counts=df["segment"].value_counts().to_dict(),
        churn_counts=df["churn"].value_counts().to_dict(),

        revenue_counts=df.groupby("segment")["purchase_amount"].sum().to_dict()
    )


# =====================================================
# CHURN (FIXED)
# =====================================================
@app.route("/churn")
def churn():

    return render_template(
        "churn.html",

        churn_counts=df["churn"].value_counts().to_dict(),

        total_customers=len(df),
        churn_customers=len(df[df["churn"] == "High Risk"]),
        safe_customers=len(df[df["churn"] != "High Risk"]),
        churn_rate=round(len(df[df["churn"] == "High Risk"]) / len(df) * 100, 2),

        customers=df.to_dict(orient="records")
    )


# =====================================================
# PREDICTION
# =====================================================
@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    result = None

    if request.method == "POST":

        age = int(request.form["age"])
        income = int(request.form["income"])
        score = int(request.form["score"])
        purchase = int(request.form["purchase"])

        pred = rf_model.predict([[age, income, score, purchase]])[0]

        result = "Premium Customer" if pred == 1 else "Regular Customer"

        prediction_history.append(result)

    return render_template(
        "prediction.html",
        result=result,
        history=prediction_history
    )


# =====================================================
# RECOMMENDATION
# =====================================================
@app.route("/recommendation")
def recommendation():

    recommendations = [
        {"segment": "Premium", "product": "iPhone 16", "category": "Electronics"},
        {"segment": "Premium", "product": "MacBook Air", "category": "Electronics"},
        {"segment": "Regular", "product": "Samsung Galaxy", "category": "Electronics"},
        {"segment": "Budget", "product": "Redmi Phone", "category": "Electronics"}
    ]

    rec_df = pd.DataFrame(recommendations)

    return render_template(
        "recommendation.html",
        recommendations=recommendations,
        category_counts=rec_df["category"].value_counts().to_dict()
    )


# =====================================================
# REPORT
# =====================================================
@app.route("/download_report")
def download_report():

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("AI Customer Report", styles["Title"]),
        Paragraph(f"Total Customers: {len(df)}", styles["Normal"]),
        Paragraph(f"Revenue: ₹{df['purchase_amount'].sum()}", styles["Normal"])
    ]

    doc.build(content)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="report.pdf")


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)