import streamlit as st
import sqlite3
import pandas as pd
import joblib
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv
import os

## Gemini API
load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ---------------- ML MODEL ----------------
MODEL_URL = "https://github.com/siddharthmahajan2477-dot/CareBridge/releases/download/v1.0/resource_prediction.pkl"
MODEL_PATH = "resource_prediction.pkl"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = joblib.load(MODEL_PATH)

# ---------------- DB ----------------
conn = sqlite3.connect("carebridge.db")

patients = pd.read_sql("SELECT * FROM patients", conn)
facilities = pd.read_sql("SELECT * FROM facilities", conn)
doctors = pd.read_sql("SELECT * FROM doctors", conn)
staff = pd.read_sql("SELECT * FROM staff", conn)
ambulances = pd.read_sql("SELECT * FROM ambulances", conn)
transfers = pd.read_sql("SELECT * FROM transfers", conn)
pharmacies = pd.read_sql("SELECT * FROM pharmacies", conn)
municipality = pd.read_sql("SELECT * FROM municipality", conn)
resource_usage = pd.read_sql("SELECT * FROM resource_usage", conn)

# ---------------- SAFE FACILITY MAP ----------------
facility_city_map = facilities[["facility_id", "city"]]
facility_district_map = facilities[["facility_id", "district"]]

# ---------------- SAFE MERGE HELP ----------------
def safe_merge(df, right, key):
    if key in df.columns:
        return df.merge(right, on=key, how="left")
    return df

# ---------------- CITY MAPPING ----------------
doctors_city = safe_merge(doctors, facility_city_map, "facility_id")
ambulances_city = safe_merge(ambulances, facility_city_map, "facility_id")

# transfers city mapping
transfers_city = transfers.merge(
    facility_city_map,
    left_on="from_facility_id",
    right_on="facility_id",
    how="left"
).rename(columns={"city": "from_city"}).drop(columns=["facility_id"])

transfers_city = transfers_city.merge(
    facility_city_map,
    left_on="to_facility_id",
    right_on="facility_id",
    how="left"
).rename(columns={"city": "to_city"}).drop(columns=["facility_id"])

# ---------------- DISTRICT MAPPING (FIXED) ----------------
doctors_district = safe_merge(doctors, facility_district_map, "facility_id")
ambulances_district = safe_merge(ambulances, facility_district_map, "facility_id")

transfers_district = transfers.merge(
    facility_district_map,
    left_on="from_facility_id",
    right_on="facility_id",
    how="left"
).rename(columns={"district": "from_district"}).drop(columns=["facility_id"])

transfers_district = transfers_district.merge(
    facility_district_map,
    left_on="to_facility_id",
    right_on="facility_id",
    how="left"
).rename(columns={"district": "to_district"}).drop(columns=["facility_id"])

# ---------------- UI ----------------
st.set_page_config(page_title="CareBridge", page_icon="🏥", layout="wide")

st.title("🏥 CareBridge")
st.subheader("AI Powered Healthcare Coordination Platform")
st.success("✅ CareBridge System Online")

# ---------------- SIDEBAR ----------------
page = st.sidebar.selectbox(
    "Navigation",
    [
        "Dashboard",
        "AI Assistant",
        "AI Reports",
        "Patients",
        "Facilities",
        "Doctors",
        "Ambulances",
        "Transfers",
        "Municipality",
        "ML Prediction"
    ]
)

##Alert system
st.subheader("🚨 Healthcare Alerts")
pending_transfers = len(
    transfers[
        transfers["transfer_status"] == "Pending"
    ]
)

if pending_transfers > 100:
    st.error(
        f"🚨 High Pending Transfers: {pending_transfers}"
    )

if resource_usage["beds_occupied"].mean() > 35:
    st.warning(
        "⚠ High Bed Occupancy Detected"
    )

if resource_usage["ambulances_used"].mean() > 2:
    st.warning(
        "🚑 High Ambulance Utilization"
    )



# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.title("Healthcare Dashboard")

    st.dataframe(patients.head())

    st.plotly_chart(
        px.histogram(facilities, x="facility_type", title="Facility Distribution"),
        use_container_width=True
    )

    st.plotly_chart(
        px.histogram(transfers, x="transfer_status", title="Transfer Status"),
        use_container_width=True
    )

    if "beds_occupied" in resource_usage.columns:
        st.plotly_chart(
            px.line(resource_usage.head(500), y="beds_occupied", title="Bed Occupancy Trend"),
            use_container_width=True
        )

# ---------------- AI ASSISTANT ----------------
if page == "AI Assistant":

    st.title("🤖 CareBridge AI Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    user_question = st.chat_input("Ask about healthcare system...")

    if user_question:

        st.session_state.chat_history.append({"role": "user", "content": user_question})

        prompt = f"""
        CareBridge AI System

        Patients: {len(patients)}
        Doctors: {len(doctors)}
        Facilities: {len(facilities)}
        Ambulances: {len(ambulances)}
        Transfers: {len(transfers)}

        Question:
        {user_question}
        """

        response = gemini_model.generate_content(prompt)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.text
        })

        with st.chat_message("assistant"):
            st.write(response.text)

    if st.button("Clear Chat"):
        st.session_state.chat_history = []

# ---------------- AI REPORTS ----------------

# =====================================================
# AI REPORTS
# =====================================================

if page == "AI Reports":

    st.title("📋 AI Municipality Report")

    cities = sorted(
        facilities["city"].dropna().unique()
    )

    selected_city = st.selectbox(
        "Select City",
        ["All Cities"] + list(cities)
    )

    # ---------------- FILTER DATA ----------------

    if selected_city == "All Cities":

        city_patients = patients
        city_doctors = doctors_city
        city_facilities = facilities
        city_ambulances = ambulances_city
        city_transfers = transfers_city
        city_resource = resource_usage

    else:

        city_patients = patients[
            patients["city"] == selected_city
        ]

        city_doctors = doctors_city[
            doctors_city["city"] == selected_city
        ]

        city_facilities = facilities[
            facilities["city"] == selected_city
        ]

        city_ambulances = ambulances_city[
            ambulances_city["city"] == selected_city
        ]

        city_transfers = transfers_city[
            (transfers_city["from_city"] == selected_city)
            |
            (transfers_city["to_city"] == selected_city)
        ]

        city_resource = resource_usage[
            resource_usage["facility_id"].isin(
                city_facilities["facility_id"]
            )
        ]

    # ---------------- BED ANALYSIS ----------------

    total_beds = city_facilities["beds_total"].sum()

    available_beds = city_facilities["beds_available"].sum()

    occupied_beds = total_beds - available_beds

    bed_utilization = 0

    if total_beds > 0:
        bed_utilization = (
            occupied_beds / total_beds
        ) * 100

    # ---------------- RATIOS ----------------

    doctor_patient_ratio = 0

    if len(city_doctors) > 0:
        doctor_patient_ratio = (
            len(city_patients)
            /
            len(city_doctors)
        )

    ambulance_patient_ratio = 0

    if len(city_ambulances) > 0:
        ambulance_patient_ratio = (
            len(city_patients)
            /
            len(city_ambulances)
        )

    # ---------------- TRANSFERS ----------------

    active_transfers = 0

    if "transfer_status" in city_transfers.columns:

        active_transfers = len(
            city_transfers[
                city_transfers["transfer_status"]
                ==
                "Pending"
            ]
        )

    # ---------------- RISK ANALYSIS ----------------

    bed_risk = "Low"

    if bed_utilization > 85:
        bed_risk = "High"

    elif bed_utilization > 70:
        bed_risk = "Medium"

    ambulance_risk = "Low"

    if ambulance_patient_ratio > 100:
        ambulance_risk = "High"

    elif ambulance_patient_ratio > 60:
        ambulance_risk = "Medium"

    transfer_risk = "Low"

    if active_transfers > 100:
        transfer_risk = "High"

    elif active_transfers > 50:
        transfer_risk = "Medium"

    # ---------------- KPI CARDS ----------------

    # ---------------- KPI CARDS ----------------
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Beds",
            total_beds
        )
    
    with col2:
        st.metric(
            "Available Beds",
            available_beds
        )
    
    with col3:
        st.metric(
            "Bed Utilization %",
            f"{bed_utilization:.1f}%"
        )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric(
            "Pending Transfers",
            active_transfers
        )
    
    with col5:
        st.metric(
            "Patient / Doctor Ratio",
            f"{doctor_patient_ratio:.1f}"
        )
    
    with col6:
        st.metric(
            "Patient / Ambulance Ratio",
            f"{ambulance_patient_ratio:.1f}"
        )
    
    st.divider()

    st.subheader("🚨 Healthcare Alerts")

    if bed_utilization > 85:
        st.error(
            f"🚨 Critical Bed Utilization: {bed_utilization:.1f}%"
        )
    elif bed_utilization > 70:
        st.warning(
            f"⚠ High Bed Utilization: {bed_utilization:.1f}%"
        )
    else:
        st.success(
            "✅ Bed Capacity Stable"
        )
    
    if active_transfers > 100:
        st.error(
            f"🚨 High Pending Transfers: {active_transfers}"
        )
    elif active_transfers > 50:
        st.warning(
            f"⚠ Transfer Backlog Detected: {active_transfers}"
        )
    else:
        st.success(
            "✅ Transfer Flow Normal"
        )
    
    if ambulance_patient_ratio > 100:
        st.error(
            f"🚨 Ambulance Shortage Risk ({ambulance_patient_ratio:.1f} patients per ambulance)"
        )
    elif ambulance_patient_ratio > 60:
        st.warning(
            f"⚠ Ambulance Pressure Increasing ({ambulance_patient_ratio:.1f} patients per ambulance)"
        )
    else:
        st.success(
            "✅ Ambulance Availability Healthy"
        )
    
    if doctor_patient_ratio > 50:
        st.error(
            f"🚨 Doctor Resource Pressure ({doctor_patient_ratio:.1f} patients per doctor)"
        )
    elif doctor_patient_ratio > 25:
        st.warning(
            f"⚠ Doctor Workload Rising ({doctor_patient_ratio:.1f} patients per doctor)"
        )
    else:
        st.success(
            "✅ Doctor Availability Adequate"
        )
    
    st.divider()

    st.subheader("📊 Quick Insights")

    st.info(
        f"""
        • Total Healthcare Facilities: {len(city_facilities)}
    
        • Total Doctors Available: {len(city_doctors)}
    
        • Total Ambulances Available: {len(city_ambulances)}
    
        • Total Patients Covered: {len(city_patients)}
    
        • Active Transfers: {active_transfers}
    
        • Bed Utilization Rate: {bed_utilization:.1f}%
        """
    )

    st.subheader("🏆 Top Facilities by Available Beds")

    top_facilities = city_facilities.sort_values(
        "beds_available",
        ascending=False
    ).head(5)
    
    st.dataframe(
        top_facilities[
            [
                "facility_name",
                "facility_type",
                "beds_total",
                "beds_available"
            ]
        ]
    )

    st.subheader("🚑 Ambulance Status")

    ambulance_status = city_ambulances["status"].value_counts()
    
    st.bar_chart(
        ambulance_status
    )

    st.subheader("🏥 Facility Distribution")

    facility_chart = px.pie(
        city_facilities,
        names="facility_type",
        title="Facility Types"
    )
    
    st.plotly_chart(
        facility_chart,
        use_container_width=True
    )

    st.subheader("📈 Resource Utilization Trend")

    # Ensure date is datetime
    city_resource["date"] = pd.to_datetime(city_resource["date"])
    
    # Sort by date (VERY IMPORTANT)
    city_resource = city_resource.sort_values("date")
    
    trend_chart = px.line(
        city_resource,
        x="date",
        y="beds_occupied",
        title="Bed Occupancy Trend Over Time"
    )
    
    st.plotly_chart(
        trend_chart,
        use_container_width=True
    )
    
    # ---------------- GENERATE REPORT ----------------

    if st.button("Generate Report"):

        avg_beds = (
            city_resource["beds_occupied"].mean()
            if len(city_resource) > 0
            else 0
        )

        peak_beds = (
            city_resource["beds_occupied"].max()
            if len(city_resource) > 0
            else 0
        )

        avg_ambulance = (
            city_resource["ambulances_used"].mean()
            if len(city_resource) > 0
            else 0
        )

        prompt = f"""
You are CareBridge AI.

Generate a professional healthcare municipality report.

CITY:
{selected_city}

DATA SUMMARY

Patients:
{len(city_patients)}

Doctors:
{len(city_doctors)}

Facilities:
{len(city_facilities)}

Ambulances:
{len(city_ambulances)}

Transfers:
{len(city_transfers)}

Pending Transfers:
{active_transfers}

BED ANALYSIS

Total Beds:
{total_beds}

Available Beds:
{available_beds}

Occupied Beds:
{occupied_beds}

Bed Utilization:
{bed_utilization:.2f}%

RESOURCE ANALYSIS

Patient Doctor Ratio:
{doctor_patient_ratio:.2f}

Patient Ambulance Ratio:
{ambulance_patient_ratio:.2f}

Average Bed Occupancy:
{avg_beds:.2f}

Peak Bed Occupancy:
{peak_beds}

Average Ambulance Usage:
{avg_ambulance:.2f}

RISK LEVELS

Bed Risk:
{bed_risk}

Ambulance Risk:
{ambulance_risk}

Transfer Risk:
{transfer_risk}

INSTRUCTIONS

Use only the supplied dataset values.

Provide:

1. Executive Summary
2. Healthcare Infrastructure Status
3. Bed Occupancy Analysis
4. Ambulance & Transfer Analysis
5. Risk Assessment
6. Resource Optimization Recommendations
7. Final AI Decision Summary
"""

        with st.spinner(
            "Generating AI Report..."
        ):

            response = gemini_model.generate_content(
                prompt
            )

            report_text = response.text

            st.write(report_text)
            
            st.download_button(
                "📥 Download Report",
                report_text,
                file_name=f"{selected_city}_healthcare_report.txt",
                mime="text/plain"
            )

# ---------------- OTHER PAGES (UNCHANGED) ----------------
if page == "Patients":
    st.title("Patients")
    st.metric("Total Patients", len(patients))
    st.metric("Active Patients", len(patients))
    search = st.text_input("Search Patient")
    st.dataframe(patients, use_container_width=True)

if page == "Facilities":
    st.title("Facilities")
    st.dataframe(facilities)
    st.metric("Total Facilities", len(facilities))
    st.metric("Total Beds", facilities["beds_total"].sum())
    st.metric("Available Beds", facilities["beds_available"].sum())
    fig = px.bar(
        facilities,
        x="facility_type",
        title="Facility Distribution"
    )
    
    st.plotly_chart(fig, use_container_width=True)

if page == "Doctors":
    st.title("Doctors")
    st.dataframe(doctors)
    st.metric("Total Doctors", len(doctors))
    doctor_city = doctors.merge(facility_city_map, on="facility_id")
    city_count = doctor_city["city"].value_counts().reset_index()
    city_count.columns = ["city", "doctors"]
    fig = px.bar(city_count, x="city", y="doctors", title="Doctors per City")
    st.plotly_chart(fig, use_container_width=True)

if page == "Ambulances":
    st.title("Ambulances")
    st.dataframe(ambulances)
    st.metric("Total Ambulances", len(ambulances))
    ambulance_city = ambulances.merge(facility_city_map, on="facility_id")
    city_amb = ambulance_city["city"].value_counts().reset_index()
    city_amb.columns = ["city", "ambulances"]
    
    fig = px.bar(city_amb, x="city", y="ambulances", title="Ambulances per City")
    st.plotly_chart(fig, use_container_width=True)

if page == "Transfers":
    st.title("Transfers")
    st.dataframe(transfers)
    st.metric("Total Transfers", len(transfers))
    st.metric("Pending Transfers", len(transfers[transfers["transfer_status"] == "Pending"]))
    fig = px.histogram(transfers, x="transfer_status", title="Transfer Status Distribution")
    st.plotly_chart(fig, use_container_width=True)

if page == "Municipality":
    st.title("Municipality")
    st.dataframe(
    municipality.drop_duplicates()
)

# ---------------- ML ----------------
if page == "ML Prediction":

    st.title("Bed Occupancy Prediction")

    inputs = [
        st.number_input("Patients In", 0, value=5),
        st.number_input("Patients Out", 0, value=4),
        st.number_input("Beds Available", 1, value=20),
        st.number_input("Ambulances Used", 0, value=1),
        st.number_input("Doctors Available", 1, value=5),
        st.number_input("Staff Available", 1, value=12)
    ]

    if st.button("Predict"):
        pred = model.predict([inputs])
        st.success(f"Predicted Beds Occupied: {pred[0]:.0f}")
