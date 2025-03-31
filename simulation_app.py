import streamlit as st
from sim_tools.distributions import Discrete
import simpy
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# Mean values from the article (group1 - group5)
LIST_FREQ_FEMALE= [1.4, 2.0, 1.6, 1.6, 2.2, 2.4, 2.2, 3.8, 3.2, 3.0, 2.4, 3.4,
                   5.0, 3.4, 4.2, 4.2, 4.4, 3.8, 5.0, 3.2, 3.0, 4.6, 3.8, 2.4,
                   4.2, 2.8, 2.4, 4.0, 2.4, 2.4, 1.8, 1.2, 1.2, 3.4, 1.0, 0.8,
                   1.8, 0.4, 0.6, 0.4, 0.8, 1.2, 0.8, 0.6, 1.0, 1.4, 0.6, 1.4,
                   0.4, 0.2, 0.2, 0.2, 0.6, 0.4, 1.0, 0.2, 0.0, 0.6, 0.2, 0.8,
                   0.4, 0.2, 0.2, 0.2, 0.4, 0.0, 0.4, 0.2, 0.0, 0.4, 0.0, 0.2,
                   0.2, 0.0, 0.0, 0.0, 0.0, 0.4, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0,
                   0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.6, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
LIST_FREQ_MALE = [0.8, 0.2, 0.6, 1.0, 0.8, 0.2, 2.2, 1.2, 2.2, 1.2, 1.2, 2.0,
                 2.0, 2.6, 2.2, 2.0, 2.6, 1.0, 1.6, 1.2, 2.8, 1.8, 1.4, 1.6,
                 1.0, 1.8, 2.4, 1.2, 2.0, 1.4, 1.0, 0.8, 1.2, 1.4, 0.6, 0.2,
                 0.8, 0.6, 1.2, 0.6, 0.4, 0.4, 1.0, 0.8, 0.4, 0.2, 0.8, 0.2,
                 0.2, 0.6, 0.0, 0.4, 0.6, 0.2, 0.4, 0.4, 0.0, 0.4, 0.0, 0.0,
                 0.2, 0.2, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.2, 0.0, 0.0, 0.4, 0.0, 0.2, 0.0, 0.2, 0.2, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
N_STREAMS = 5
DEFAULT_RNG_SET = 12345
LoS_SEED = 43

class Scenario:
    def __init__(
        self,
        random_number_set,
        num_beds,
        num_single_rooms,
        num_multi_bed_bays,
        bay_capacity,
        mean_interarrival_time,
        probability_female,
        los_seed,
        list_freq_female,
        list_freq_male,
    ):
        # resource counts
        self.num_beds = num_beds
        self.num_single_rooms = num_single_rooms
        self.num_multi_bed_bays = num_multi_bed_bays
        self.bay_capacity = bay_capacity

        self.mean_interarrival_time = mean_interarrival_time
        self.probability_female = probability_female
        self.list_freq_female = list_freq_female
        self.list_freq_male = list_freq_male

        self.random_number_set = random_number_set
        self.los_seed = los_seed
        self.init_sampling()

    def set_random_no_set(self, random_number_set):
        """
        Controls the random sampling

        Parameters:
        ----------
        random_number_set: int
            Used to control the set of pseudo-random numbers
            used by the distributions in the simulation.
        """
        self.random_number_set = random_number_set
        self.init_sampling()

    def init_sampling(self):
        """
        Create the distributions used by the model and initialise
        the random seeds of each.
        """
        # create random number streams
        rng_streams = np.random.default_rng(self.random_number_set)
        self.seeds = rng_streams.integers(0, 999999999, size=N_STREAMS)

        # create distributions
        self.rng_ar = np.random.default_rng(seed=self.seeds[0])
        self.rng_gender = np.random.default_rng(seed=self.seeds[1])
        self.rng_los = np.random.default_rng(seed=self.los_seed)  # Use los_seed

        # LoS Discrete distribution that follows empirical values from the article / mean value of all groups.
        # It's also applicable for length of stay when not in bay. (single rooms)
        los_dist = Discrete(
            values=list(range(1, 201)),
            freq=self.list_freq_female,
            random_seed=self.rng_los.integers(0, 999999999),
        )
        self.LoS_FEMALE_S = los_dist.sample(size=1000)
        los_dist = Discrete(
            values=list(range(1, 201)),
            freq=self.list_freq_male,
            random_seed=self.rng_los.integers(0, 999999999),
        )
        self.LoS_MALE_S = los_dist.sample(size=1000)


# Patient class
class Patient:
    def __init__(self, patient_id, gender, length_of_stay):
        self.patient_id = patient_id
        self.gender = gender
        self.length_of_stay = length_of_stay
        self.arrival_time = None  # Track arrival time for waiting time calculation
        self.admission_time = None  # Track admission time for time spent on the ward
        self.accumulated_waiting_time = 0  # Initialize accumulated waiting time


# Ward class
class Ward:
    def __init__(
        self, env, num_beds, num_single_rooms, num_multi_bed_bays, bay_capacity, warm_up
    ):
        self.env = env
        self.num_beds = num_beds
        self.num_single_rooms = num_single_rooms
        self.num_multi_bed_bays = num_multi_bed_bays
        self.bay_capacity = bay_capacity
        self.warm_up = warm_up

        # Create occupancy dictionary based on NUM_MULTI_BED_BAYS
        self.occupancy = {
            f"Bay_{i}": [None] * bay_capacity for i in range(num_multi_bed_bays)
        }
        self.occupancy["Single_Rooms"] = [None] * num_single_rooms

        self.waiting_list = []
        self.metrics = {
            "total_patients": 0,
            "patients_admitted": 0,
            "total_waiting_time": 0,
            "average_waiting_time": 0,
            "female_count": 0,
            "male_count": 0,
            "bed_utilization": 0,
            "average_time_spent_on_ward": 0,
            "average_number_of_patients_on_ward": 0,
            "current_patients_on_ward": 0,
            "total_time_spent_on_ward": 0,
            "total_transfers": 0,
            "patients_left_waiting": 0,
            "queue_sizes_per_day": [],  # Queue sizes per day
            "bed_occupancy_per_day": [],  # Bed occupancy per day
        }

    def admit_patient(self, patient):
        """Attempt to allocate a bed to the patient."""
        yield self.env.timeout(0)
        self.metrics["total_patients"] += 1

        # Track gender counts
        if patient.gender == "Female":
            self.metrics["female_count"] += 1
        else:
            self.metrics["male_count"] += 1

        allocated = self.allocate_bed(patient)
        if allocated:
            self.metrics["patients_admitted"] += 1
            patient.admission_time = self.env.now
            waiting_time = (
                self.env.now - patient.arrival_time + patient.accumulated_waiting_time
            )
            self.metrics["total_waiting_time"] += waiting_time
            self.calculate_metrics()  # Recalculate metrics after changes
            self.metrics["current_patients_on_ward"] += 1
            self.env.process(self.discharge_patient(patient))
        else:
            self.waiting_list.append(patient)

            print(
                f"[Time {self.env.now:>5}] Patient {patient.patient_id:>3} "
                f"({patient.gender}) added to waiting list. "
                f"Waiting patients: {len(self.waiting_list)}"
            )
            print(
                f"[Time {self.env.now:>5}] Waiting List: "
                f"{[p.patient_id for p in self.waiting_list]}"
            )
            # Start a process to check if the patient leaves after max_waiting_time
            self.env.process(self.check_patient_leaving(patient))

    def allocate_bed(self, patient):
        """Bed allocation logic."""
        print(
            f"[Time {self.env.now:>5}] Attempting to allocate bed for Patient {patient.patient_id:>3} ({patient.gender})"
        )
        # Step 1: Assign to a multi-bed bay with the same gender
        if self.allocate_to_bay_same_gender(patient):
            return True

        # Step 2: Assign to an empty multi-bed bay
        if self.allocate_to_empty_bay(patient):
            return True

        # Step 3: Assign to a single room
        if self.allocate_single_room(patient):
            return True

        # Step 4: Move a patient of the opposite gender from a single room to a bay
        if self.move_patient_from_single_room(patient):
            return True

        # Step 5: No beds available
        print(
            f"[Time {self.env.now:>5}] No beds available for Patient {patient.patient_id:>3} ({patient.gender})"
        )
        return False

    def allocate_to_bay_same_gender(self, patient):
        """Attempt to allocate patient to a bay with same gender patients."""
        print(
            f"[Time {self.env.now:>5}] Checking for bay with same gender for Patient {patient.patient_id:>3} ({patient.gender})"
        )
        for bay_key, bay_occupancy in self.occupancy.items():
            if bay_key.startswith("Bay_"):
                if any(bay_occupancy) and all(
                    p.gender == patient.gender for p in bay_occupancy if p
                ):
                    try:
                        empty_index = bay_occupancy.index(None)
                        bay_occupancy[empty_index] = patient
                        print(
                            f"[Time {self.env.now:>5}] Patient {patient.patient_id:>3} "
                            f"({patient.gender}) assigned to {bay_key}, Bed {empty_index + 1}"
                        )
                        return True
                    except ValueError:
                        continue
        print(
            f"[Time {self.env.now:>5}] No suitable bay with same gender found for Patient {patient.patient_id:>3} ({patient.gender})"
        )
        return False

    def allocate_to_empty_bay(self, patient):
        """Attempt to allocate patient to an empty bay."""
        print(
            f"[Time {self.env.now:>5}] Checking for empty bay for Patient {patient.patient_id:>3} ({patient.gender})"
        )
        for bay_key, bay_occupancy in self.occupancy.items():
            if bay_key.startswith("Bay_"):
                if not any(bay_occupancy):  # Check if the bay is empty
                    bay_occupancy[0] = patient  # Assign to the first bed
                    print(
                        f"[Time {self.env.now:>5}] Patient {patient.patient_id:>3} "
                        f"({patient.gender}) assigned to empty {bay_key}, Bed 1"
                    )
                    return True
        print(
            f"[Time {self.env.now:>5}] No empty bay found for Patient {patient.patient_id:>3} ({patient.gender})"
        )
        return False

    def allocate_single_room(self, patient):
        """Attempt to allocate patient to a single room."""
        print(
            f"[Time {self.env.now:>5}] Checking for single room for Patient {patient.patient_id:>3} ({patient.gender})"
        )
        try:
            empty_index = self.occupancy["Single_Rooms"].index(None)
            self.occupancy["Single_Rooms"][empty_index] = patient
            print(
                f"[Time {self.env.now:>5}] Patient {patient.patient_id:>3} "
                f"({patient.gender}) assigned to Single Room {empty_index + 1}"
            )
            return True
        except ValueError:
            print(
                f"[Time {self.env.now:>5}] No single room found for Patient {patient.patient_id:>3} ({patient.gender})"
            )
            return False

    def move_patient_from_single_room(self, patient):
        """Attempt to move a patient from a single room to a bay to accommodate the current patient."""
        print(
            f"[Time {self.env.now:>5}] Checking to move patient from single room for Patient {patient.patient_id:>3} ({patient.gender})"
        )
        for i, single_room_patient in enumerate(self.occupancy["Single_Rooms"]):
            if single_room_patient and single_room_patient.gender != patient.gender:
                for bay_key, bay_occupancy in self.occupancy.items():
                    if bay_key.startswith("Bay_"):  # Only check multi-bed bays
                        if any(bay_occupancy) and all(
                            p.gender == single_room_patient.gender
                            for p in bay_occupancy
                            if p
                        ):
                            try:
                                empty_index = bay_occupancy.index(None)
                                # Before the move
                                print(
                                    f"[Time {self.env.now:>5}] Before move: {bay_key} occupancy: {bay_occupancy}, Single Room {i + 1} occupancy: {self.occupancy['Single_Rooms'][i]}"
                                )
                                bay_occupancy[
                                    empty_index
                                ] = single_room_patient  # Move patient
                                self.occupancy["Single_Rooms"][
                                    i
                                ] = patient  # Assign new patient
                                self.metrics[
                                    "total_transfers"
                                ] += 1  # Increment transfer count
                                # After the move
                                print(
                                    f"[Time {self.env.now:>5}] After move: {bay_key} occupancy: {bay_occupancy}, Single Room {i + 1} occupancy: {self.occupancy['Single_Rooms'][i]}"
                                )
                                print(
                                    f"[Time {self.env.now:>5}] Patient {patient.patient_id:>3} "
                                    f"({patient.gender}) assigned to Single Room {i + 1} after "
                                    f"moving Patient {single_room_patient.patient_id:>3} to "
                                    f"{bay_key}, Bed {empty_index + 1}"
                                )
                                return True
                            except ValueError:
                                continue  # No space in this bay, try next
        print(
            f"[Time {self.env.now:>5}] No patient moved from single room for Patient {patient.patient_id:>3} ({patient.gender})"
        )
        return False

    def discharge_patient(self, patient):
        """Discharge a patient after their length of stay."""
        yield self.env.timeout(patient.length_of_stay)  # Wait for length of stay
        # Update metrics for length of stay and time spent on the ward
        time_spent_on_ward = self.env.now - patient.admission_time
        self.metrics["total_time_spent_on_ward"] += time_spent_on_ward

        # Find and release the patient from a bay
        for bay_key, bay_occupancy in self.occupancy.items():
            if bay_occupancy is not None:
                try:
                    patient_index = bay_occupancy.index(patient)
                    bay_occupancy[patient_index] = None
                    print(
                        f"[Time {self.env.now:>5}] Patient {patient.patient_id:>3} "
                        f"({patient.gender}) discharged from {bay_key}"
                    )
                    break
                except ValueError:
                    continue
        self.calculate_metrics()  # Update metrics after discharge

    def check_patient_leaving(self, patient):
        """Check if a patient leaves after waiting too long."""
        yield self.env.timeout(MAX_WAITING_TIME)
        if patient in self.waiting_list:
            self.waiting_list.remove(patient)
            # Accumulate waiting time before removing the patient
            waiting_time = (
                self.env.now - patient.arrival_time + patient.accumulated_waiting_time
            )
            self.metrics["total_waiting_time"] += waiting_time
            self.metrics["patients_left_waiting"] += 1  # Increment the counter
            print(
                f"[Time {self.env.now:>5}] Patient {patient.patient_id:>3} "
                f"({patient.gender}) left after waiting too long ({waiting_time:.2f})"
            )
            self.calculate_metrics()

    def print_ward_state(self):
        """Print the current state of the ward."""
        print(f"\n[Time {self.env.now:>5}] Ward State:")
        for bay_key, bay_occupancy in self.occupancy.items():
            print(
                f"{bay_key}: {[f'Patient {p.patient_id} ({p.gender})' if p else 'Empty' for p in bay_occupancy]}"
            )
        print(
            f"Waiting List: {[f'Patient {p.patient_id} ({p.gender})' for p in self.waiting_list]}\n"
        )

    def calculate_metrics(self):
        """Calculate and update the key performance indicators (KPIs)."""
        if self.env.now >= self.warm_up:  # Only calculate after warm-up
            # Average Waiting Time (only for patients who waited)
            if self.metrics["total_patients"] - len(self.waiting_list) > 0:
                self.metrics["average_waiting_time"] = self.metrics[
                    "total_waiting_time"
                ] / (self.metrics["total_patients"] - len(self.waiting_list))
            else:
                self.metrics["average_waiting_time"] = 0
            # Average Time Spent on Ward
            if self.metrics["patients_admitted"] > 0:
                self.metrics["average_time_spent_on_ward"] = (
                    self.metrics["total_time_spent_on_ward"]
                    / self.metrics["patients_admitted"]
                )
            else:
                self.metrics["average_time_spent_on_ward"] = 0

            # Average Number of Patients on Ward
            if self.env.now > 0:
                self.metrics["average_number_of_patients_on_ward"] = (
                    self.metrics["total_time_spent_on_ward"] / self.env.now
                )
            else:
                self.metrics["average_number_of_patients_on_ward"] = 0

            # Bed Utilization (average percentage)
            total_beds = (
                self.num_single_rooms + self.num_multi_bed_bays * self.bay_capacity
            )
            self.metrics["bed_utilization"] = (
                (total_beds - self.get_available_beds()) / total_beds * 100
            )

    def get_available_beds(self):
        """Calculate the number of available beds."""
        available_beds = 0
        for bay in self.occupancy.values():
            available_beds += bay.count(None)
        return available_beds


# Patient arrival process
def patient_arrival(env, ward, scenario):
    patient_id = 0
    while True:
        interarrival_time = scenario.rng_ar.exponential(
            scale=scenario.mean_interarrival_time
        )
        yield env.timeout(interarrival_time)
        patient_id += 1
        gender = (
            "Female"
            if scenario.rng_gender.binomial(n=1, p=scenario.probability_female) == 1
            else "Male"
        )
        if gender == "Female":
            length_of_stay = scenario.rng_los.choice(scenario.LoS_FEMALE_S)
        if gender == "Male":
            length_of_stay = scenario.rng_los.choice(scenario.LoS_MALE_S)
        patient = Patient(patient_id, gender, length_of_stay)
        patient.arrival_time = env.now
        print(
            f"[Time {env.now:>5}] Patient {patient.patient_id:>3} ({gender}) "
            f"arrived with length of stay {length_of_stay}"
        )
        env.process(ward.admit_patient(patient))


def periodic_ward_checks(env, ward, interval=1):
    while True:
        yield env.timeout(interval)
        if env.now >= ward.warm_up:
            ward.calculate_metrics()
            ward.metrics["queue_sizes_per_day"].append(len(ward.waiting_list))
            total_beds = (
                ward.num_single_rooms + ward.num_multi_bed_bays * ward.bay_capacity
            )
            occupied_beds = total_beds - ward.get_available_beds()
            occupancy_percentage = (
                (occupied_beds / total_beds) * 100 if total_beds > 0 else 0
            )
            ward.metrics["bed_occupancy_per_day"].append(occupancy_percentage)
            ward.print_ward_state()

def single_run(scenario, sim_time, random_no_set, warm_up):
    """
    Perform a single run of the ward simulation and return the metrics.
    """
    # Set random number set
    scenario.set_random_no_set(random_no_set)

    # Simulation Setup
    env = simpy.Environment()
    ward = Ward(
        env,
        scenario.num_beds,
        scenario.num_single_rooms,
        scenario.num_multi_bed_bays,
        scenario.bay_capacity,
        warm_up=warm_up,
    )
    env.process(patient_arrival(env, ward, scenario))
    env.process(periodic_ward_checks(env, ward))
    # Run the simulation
    env.run(until=sim_time)

    # Calculate final metrics to ensure accuracy
    ward.calculate_metrics()  # Ensure final metrics are calculated
    return ward.metrics


def multiple_replications(scenario, sim_time, warm_up, n_reps, n_jobs=-1):
    """
    Perform multiple replications of the ward simulation.
    """
    res = Parallel(n_jobs=n_jobs)(
        delayed(single_run)(scenario, sim_time, random_no_set=rep, warm_up=warm_up)
        for rep in range(n_reps)
    )

    # Format and return results in a DataFrame
    df_results = pd.DataFrame(res)
    df_results.index = np.arange(1, len(df_results) + 1)
    df_results.index.name = "rep"
    return df_results
st.title("Hospital Ward Simulation")
st.write("This app simulates patient flow in a hospital ward. Adjust the parameters below.")

# Sidebar - Input Parameters
st.sidebar.header("Simulation Parameters")

# Ward Configuration
num_beds = st.sidebar.number_input("Number of Beds", min_value=1, value=15)
num_single_rooms = st.sidebar.number_input("Number of Single Rooms", min_value=0, value=2)
num_multi_bed_bays = st.sidebar.number_input("Number of Multi-Bed Bays", min_value=0, value=3)
bay_capacity = st.sidebar.number_input("Bay Capacity", min_value=1, value=4)

# Validation
total_bay_beds = num_multi_bed_bays * bay_capacity
total_required_beds = num_single_rooms + total_bay_beds

if total_required_beds != num_beds:
    st.sidebar.error(f"Error: Total required beds ({total_required_beds}) exceed the available beds ({num_beds}).")
    st.sidebar.info("Please adjust the number of single rooms or multi-bed bays to ensure the total required beds do not exceed the available beds.")
    st.stop()  # Stop the app if validation fails
    
# Patient Arrival
mean_interarrival_time = st.sidebar.number_input("Mean Interarrival Time", min_value=0.1, value=1/0.774822517748225)
probability_female = st.sidebar.number_input("Probability of Female Patient", min_value=0.0, max_value=1.0, value=0.661376)

# Simulation Control
sim_time = st.sidebar.number_input("Simulation Time", min_value=1, value=1000)
warm_up = st.sidebar.number_input("Warm-up Period", min_value=0, value=100)
n_reps = st.sidebar.number_input("Number of Replications", min_value=1, value=93)
MAX_WAITING_TIME = st.sidebar.number_input("Max Waiting Time", min_value=1, value=10)

# Run Simulation Button
if st.sidebar.button("Run Simulation"):
    # Convert string inputs to lists of floats
    list_freq_female = LIST_FREQ_FEMALE
    list_freq_male = LIST_FREQ_MALE

    # Create Scenario
    scenario = Scenario(
        random_number_set=DEFAULT_RNG_SET,
        num_beds=num_beds,
        num_single_rooms=num_single_rooms,
        num_multi_bed_bays=num_multi_bed_bays,
        bay_capacity=bay_capacity,
        mean_interarrival_time=mean_interarrival_time,
        probability_female=probability_female,
        los_seed=LoS_SEED,
        list_freq_female=list_freq_female,
        list_freq_male=list_freq_male,
    )
    # Run Simulation
    with st.spinner("Running Simulation..."):
        results_df = multiple_replications(
            scenario, sim_time=sim_time, warm_up=warm_up, n_reps=n_reps, n_jobs=-1
        )

    # Display Results
    st.header("Simulation Results")
    st.dataframe(results_df)

    # Display key metrics more clearly
    st.subheader("Key Metrics (Averaged over Replications)")
    mean_results = results_df.iloc[:, :-2].mean()
    st.write(f"Average Waiting Time: {mean_results['average_waiting_time']:.2f}")
    st.write(f"Average Bed Utilization: {mean_results['bed_utilization']:.2f}%")
    st.write(f"Average Time Spent on Ward: {mean_results['average_time_spent_on_ward']:.2f}")
    st.write(f"Average Number of Patients on Ward: {mean_results['average_number_of_patients_on_ward']:.2f}")
    st.write(f"Total Patients Left")

    # Visualizations
    st.subheader("Visualizations")
    st.line_chart(results_df['average_waiting_time'])
    st.bar_chart(results_df['bed_utilization'])
