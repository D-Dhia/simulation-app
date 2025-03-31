import streamlit as st
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(
    page_title="Hospital Ward Simulation",
    page_icon="🏥",
    layout="wide"
)

# Title and Introduction
st.title("🏥 Hospital Ward Patient Flow Simulation")
st.markdown("""
Simulate patient flow in a hospital ward with single rooms and multi-bed bays. Adjust parameters to observe effects on patient waiting times, bed occupancy, and utilization.
""")

# Sidebar: Interactive Parameters
st.sidebar.header("⚙️ Simulation Parameters")
SIM_TIME = st.sidebar.slider("Simulation Duration (days)", 50, 500, 100, step=10)
NUM_BEDS = st.sidebar.slider("Total Beds", 10, 50, 20)
NUM_SINGLE_ROOMS = st.sidebar.slider("Single Rooms", 1, 10, 5)
NUM_MULTI_BED_BAYS = st.sidebar.slider("Multi-Bed Bays", 1, 10, 3)
BAY_CAPACITY = st.sidebar.slider("Beds per Bay", 2, 10, 5)
ARRIVAL_RATE = st.sidebar.slider("Arrival Rate (patients/day)", 0.1, 2.0, 0.5, step=0.1)

# Constants for patient characteristics
PROBABILITY_FEMALE = 0.661376
LEFT, MODE, RIGHT = 1, 5, 155
MAX_WAITING_TIME = 10

# Random number generators for reproducibility
rng_los = np.random.default_rng(seed=43)
rng_gender = np.random.default_rng(seed=53)
rng_ar = np.random.default_rng(seed=63)

# Patient Class Definition
class Patient:
    def __init__(self, patient_id, gender, length_of_stay):
        self.patient_id = patient_id
        self.gender = gender
        self.length_of_stay = length_of_stay
        self.arrival_time = None
        self.admission_time = None

# Ward Class Definition
class Ward:
    def __init__(self, env, num_beds, num_single_rooms, num_multi_bed_bays, bay_capacity):
        self.env = env
        self.num_beds = num_beds
        self.num_single_rooms = num_single_rooms
        self.num_multi_bed_bays = num_multi_bed_bays
        self.bay_capacity = bay_capacity

        self.occupancy = {f"Bay_{i}": [None] * bay_capacity for i in range(num_multi_bed_bays)}
        self.occupancy["Single_Rooms"] = [None] * num_single_rooms

        self.waiting_list = []
        self.metrics = {
            "total_patients": 0,
            "patients_admitted": 0,
            "patients_waiting": 0,
            "patients_who_waited": 0,
            "patients_left": 0,
            "total_waiting_time": 0,
            "queue_sizes": [],
            "occupancy_percentages": [],
            "num_transfers": 0,
            "max_waiting_time": 0,
        }

    def admit_patient(self, patient):
        yield self.env.timeout(0)
        allocated = self.allocate_bed(patient)
        if allocated:
            patient.admission_time = self.env.now
            self.metrics["patients_admitted"] += 1
            self.env.process(self.discharge_patient(patient))
        else:
            self.waiting_list.append(patient)
            self.metrics["patients_waiting"] += 1

    def allocate_bed(self, patient):
        for bay_key in self.occupancy:
            if None in self.occupancy[bay_key]:
                empty_index = self.occupancy[bay_key].index(None)
                self.occupancy[bay_key][empty_index] = patient
                return True
        return False

    def discharge_patient(self, patient):
        yield self.env.timeout(patient.length_of_stay)
        for bay_key in self.occupancy:
            if patient in self.occupancy[bay_key]:
                index = self.occupancy[bay_key].index(patient)
                self.occupancy[bay_key][index] = None

    def update_metrics(self):
        occupied_beds = sum(1 for bay in self.occupancy.values() for bed in bay if bed is not None)
        occupancy_percentage = (occupied_beds / self.num_beds) * 100
        self.metrics["queue_sizes"].append(len(self.waiting_list))
        self.metrics["occupancy_percentages"].append(occupancy_percentage)

# Patient Arrival Process
def patient_arrival(env, ward):
    patient_id = 0
    while True:
        yield env.timeout(rng_ar.poisson(lam=ARRIVAL_RATE))
        patient_id += 1
        gender = 'Female' if rng_gender.binomial(n=1, p=PROBABILITY_FEMALE) else 'Male'
        length_of_stay = rng_los.triangular(LEFT, MODE, RIGHT)
        patient = Patient(patient_id, gender, length_of_stay)
        env.process(ward.admit_patient(patient))

# Periodic Metrics Update Process
def periodic_metrics(env, ward):
    while True:
        yield env.timeout(1) # daily updates
        ward.update_metrics()

# Run Simulation Button
if st.button("▶️ Run Simulation"):
    with st.spinner("Running simulation..."):
        env = simpy.Environment()
        ward = Ward(env, NUM_BEDS, NUM_SINGLE_ROOMS, NUM_MULTI_BED_BAYS, BAY_CAPACITY)

        env.process(patient_arrival(env, ward))
        env.process(periodic_metrics(env, ward))

        env.run(until=SIM_TIME)

    st.success("Simulation completed successfully!")

    days = np.arange(len(ward.metrics["queue_sizes"]))
    df_metrics = pd.DataFrame({
        "Day": days,
        "Queue Size": ward.metrics["queue_sizes"],
        "Occupancy (%)": ward.metrics["occupancy_percentages"]
    })

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Queue Size Over Time")
        plt.figure(figsize=(10,4))
        sns.lineplot(data=df_metrics,x="Day",y="Queue Size",color="purple")
        plt.grid(True)
        st.pyplot(plt.gcf())

    with col2:
        st.subheader("🛏️ Bed Occupancy (%) Over Time")
        plt.figure(figsize=(10,4))
        sns.lineplot(data=df_metrics,x="Day",y="Occupancy (%)",color="green")
        plt.fill_between(df_metrics["Day"],df_metrics["Occupancy (%)"],alpha=0.3,color="green")
        plt.grid(True)
        st.pyplot(plt.gcf())

else:
    st.info('👈 Adjust parameters in the sidebar and click "Run Simulation" to start.')
