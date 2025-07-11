import streamlit as st
import simpy
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.title("Simulation de la logistique de pose d'enrobés")
st.divider()
st.header("1. Simulation du mouvement des camions")


col1, col2 = st.columns(2)
with col1:
    st.subheader("Donées de base")
    c_poste = st.number_input(
        "Capacité du poste d'enrobé (tonnes/heure)", min_value=50, value=180
    )
    c_camions = st.number_input("Capacité de camions (tonnes)", min_value=5, value=40)
    d_bache = st.number_input("Durée de bâchage (minutes)", min_value=1, value=8)
    t_dechargement = st.number_input(
        "Temps de déchargement (minutes)", min_value=1, value=8
    )
    v_allez = st.slider("Interval vitesse allez(camions)", 0, 100, (15, 25), 5)
with col2:
    st.subheader("Donées de production")
    n_camion = st.number_input("Nombre du camions", min_value=1, value=10)
    distance = st.number_input(
        "Distance entre le poste d'enrobé et l'atelier (mètres)",
        min_value=500,
        value=1000,
    )
    q_cible = st.number_input(
        "Quantité cible d'enrobé à produire/poser (tonnes)", min_value=40, value=1000
    )
    n_start = st.number_input(
        "Nombre de camions a stockeravant de commencer la pose",
        min_value=1,
        value=1,
    )
    v_retour = st.slider("Interval vitesse retour (camions)", 0, 100, (30, 40), 5)

min_a, max_a = v_allez
min_r, max_r = v_retour


# setup of the simulation
class AsphaltSimulation:
    def __init__(
        self,
        num_trucks,
        distance_plant_to_machine,
        target_quantity,
        trucks_required_to_start_laying=1,
    ):
        """
        Initialiser la simulation de la pose d'enrobé.

        Arguments :
            num_trucks: Nombre de camions dans la simulation
            distance_plant_to_machine: Distance entre le poste d'enrobé et l'atelier (mètres)
            target_quantity: Quantité cible d'enrobé à produire/poser (tonnes)
            trucks_required_to_start_laying: Nombre de camions devant arriver à l'atelier avant de commencer la pose
        """
        self.env = simpy.Environment()
        self.num_trucks = num_trucks
        self.distance = distance_plant_to_machine
        self.target_quantity = target_quantity
        self.trucks_required_to_start_laying = trucks_required_to_start_laying

        # Ressources
        self.plant_loader = simpy.Resource(self.env, capacity=1)
        self.machine_unloader = simpy.Resource(self.env, capacity=1)

        # Constantes
        self.PLANT_CAPACITY = c_poste  # tonnes par heure
        self.TRUCK_CAPACITY = c_camions  # tonnes
        self.DUREE_BACHAGE = d_bache  # minutes (5 minutes pour charger 40 tonnes)
        self.LOADING_TIME = (
            self.TRUCK_CAPACITY / self.PLANT_CAPACITY
        ) * 60 + self.DUREE_BACHAGE  # minutes
        self.UNLOADING_TIME = t_dechargement  # minutes (40 tonnes en 10 minutes)
        self.MACHINE_SPEED = 2.5  # mètres par seconde

        # Plages de vitesses (km/h)
        self.LOADED_SPEED_RANGE = (min_a, max_a)
        self.EMPTY_SPEED_RANGE = (min_r, max_r)

        # Variables de suivi
        self.stats = {
            "trucks_at_plant": [],
            "trucks_at_machine": [],
            "trucks_on_road": [],
            "time_stamps": [],
            "total_asphalt_delivered": 0,
            "total_asphalt_laid": 0,
            "truck_utilization": [],
            "plant_utilization": 0,
            "machine_utilization": 0,
        }

        # Suivi de la production du poste d'enrobé
        self.plant_production = 0
        self.plant_start_time = 0
        self.machine_laying_time = 0

        # Contrôle de la simulation
        self.simulation_done = (
            self.env.event()
        )  # Événement pour signaler la fin de la simulation
        self.trucks_unloaded = 0

        # Pour la logique de démarrage de la pose
        self.trucks_waiting_at_machine = []
        self.laying_started = False

    def kmh_to_ms(self, speed_kmh):
        """Convertir km/h en m/s"""
        return speed_kmh / 3.6

    def calculate_travel_time(self, distance, is_loaded):
        """Calculer le temps de trajet selon la distance et l'état de charge"""
        if is_loaded:
            speed_kmh = random.uniform(*self.LOADED_SPEED_RANGE)
        else:
            speed_kmh = random.uniform(*self.EMPTY_SPEED_RANGE)

        speed_ms = self.kmh_to_ms(speed_kmh)
        travel_time_seconds = distance / speed_ms
        return travel_time_seconds / 60  # Conversion en minutes

    def plant_process(self):
        """Simuler la production du poste d'enrobé"""
        while True:
            # Vérifier si le poste peut produire (160 t/h = 2,67 t/min)
            production_rate = self.PLANT_CAPACITY / 60  # tonnes par minute
            yield self.env.timeout(1)  # Vérifier chaque minute

            # Suivi de la production
            if self.plant_production < self.PLANT_CAPACITY * (self.env.now / 60):
                self.plant_production += production_rate
            # Vérifier si la quantité cible est atteinte
            if (
                self.target_quantity is not None
                and self.plant_production >= self.target_quantity
            ):
                break

    def truck_process(self, truck_id):
        """Simuler le cycle d'un camion"""
        while True:
            # 1. Aller au poste d'enrobé (à vide)
            travel_time = self.calculate_travel_time(self.distance, is_loaded=False)
            yield self.env.timeout(travel_time)

            # 2. Attendre et charger au poste d'enrobé
            with self.plant_loader.request() as request:
                yield request
                self.plant_start_time = self.env.now
                # Si la cible est atteinte, ne pas charger plus
                if (
                    self.target_quantity is not None
                    and self.stats["total_asphalt_delivered"] >= self.target_quantity
                ):
                    break
                yield self.env.timeout(self.LOADING_TIME)
                self.stats["total_asphalt_delivered"] += self.TRUCK_CAPACITY
                # Si après chargement la cible est atteinte, marquer pour la dernière livraison
                if (
                    self.target_quantity is not None
                    and self.stats["total_asphalt_delivered"] >= self.target_quantity
                ):
                    self.last_truck = truck_id
            # 3. Aller à l'atelier (chargé)
            travel_time = self.calculate_travel_time(self.distance, is_loaded=True)
            yield self.env.timeout(travel_time)
            # 4. Attendre à l'atelier jusqu'à ce qu'il y ait assez de camions
            self.trucks_waiting_at_machine.append(truck_id)
            while not self.laying_started:
                if (
                    len(self.trucks_waiting_at_machine)
                    >= self.trucks_required_to_start_laying
                ):
                    self.laying_started = True
                else:
                    yield self.env.timeout(1)  # Attendre 1 minute et vérifier à nouveau
            # 5. Attendre et décharger à l'atelier
            with self.machine_unloader.request() as request:
                yield request
                machine_start_time = self.env.now
                yield self.env.timeout(self.UNLOADING_TIME)
                # Ajouter le minimum entre la capacité du camion et le reste à livrer
                remaining = (
                    self.target_quantity - self.stats["total_asphalt_laid"]
                    if self.target_quantity is not None
                    else self.TRUCK_CAPACITY
                )
                delivered = min(self.TRUCK_CAPACITY, remaining)
                self.stats["total_asphalt_laid"] += delivered
                self.machine_laying_time += self.UNLOADING_TIME
                # Retirer le camion de la liste d'attente
                if truck_id in self.trucks_waiting_at_machine:
                    self.trucks_waiting_at_machine.remove(truck_id)
                # Si c'est le dernier camion, incrémenter le compteur
                if (
                    self.target_quantity is not None
                    and hasattr(self, "last_truck")
                    and truck_id == self.last_truck
                ):
                    self.trucks_unloaded += 1
                    if self.trucks_unloaded == 1:
                        self.simulation_done.succeed()
                    break
            # Si la cible est atteinte, arrêter le cycle
            if (
                self.target_quantity is not None
                and self.stats["total_asphalt_delivered"] >= self.target_quantity
            ):
                break

    def monitor_system(self, time_step=5):
        """Surveiller l'état du système à intervalles réguliers"""
        while True:
            # Compter les camions à différents endroits
            trucks_at_plant = len(self.plant_loader.queue) + len(
                self.plant_loader.users
            )
            trucks_at_machine = len(self.machine_unloader.queue) + len(
                self.machine_unloader.users
            )
            trucks_on_road = self.num_trucks - trucks_at_plant - trucks_at_machine

            # Enregistrer les statistiques
            self.stats["time_stamps"].append(self.env.now)
            self.stats["trucks_at_plant"].append(trucks_at_plant)
            self.stats["trucks_at_machine"].append(trucks_at_machine)
            self.stats["trucks_on_road"].append(trucks_on_road)

            yield self.env.timeout(time_step)

    def run_simulation(self, time_step=5):
        """Lancer la simulation complète"""
        # Démarrer le poste d'enrobé
        self.env.process(self.plant_process())

        # Démarrer les processus camions
        for i in range(self.num_trucks):
            self.env.process(self.truck_process(i))

        # Démarrer la surveillance
        self.env.process(self.monitor_system(time_step))

        # Lancer la simulation jusqu'à la quantité cible
        self.env.run(until=self.simulation_done)

        # Calculer les statistiques finales
        self.calculate_final_stats()

    def calculate_final_stats(self):
        """Calculer les statistiques finales de la simulation"""
        if len(self.stats["time_stamps"]) > 0:
            total_time_minutes = self.stats["time_stamps"][-1]
        else:
            total_time_minutes = 0
        total_time_hours = total_time_minutes / 60
        theoretical_production = self.PLANT_CAPACITY * total_time_hours
        self.stats["plant_utilization"] = (
            (self.stats["total_asphalt_delivered"] / theoretical_production) * 100
            if theoretical_production > 0
            else 0
        )
        self.stats["machine_utilization"] = (
            (self.machine_laying_time / total_time_minutes) * 100
            if total_time_minutes > 0
            else 0
        )
        avg_working_trucks = (
            np.mean(
                [
                    p + m
                    for p, m in zip(
                        self.stats["trucks_at_plant"], self.stats["trucks_at_machine"]
                    )
                ]
            )
            if self.stats["trucks_at_plant"]
            else 0
        )
        self.stats["truck_utilization"] = (
            (avg_working_trucks / self.num_trucks) * 100 if self.num_trucks > 0 else 0
        )


# Initialiser le session state pour persister les résultats
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None


def run_sim():
    """Fonction pour lancer la simulation et afficher les résultats"""
    sim = AsphaltSimulation(
        n_camion, distance, q_cible, n_start
    )  # Initialiser la simulation
    sim.run_simulation()  # Lancer la simulation

    # Stocker les résultats dans session state
    st.session_state.simulation_results = sim.stats
    st.success("Simulation terminée avec succès!")


if st.button("Lancer la simulation"):
    with st.spinner("Simulation en cours..."):
        run_sim()

# Afficher les résultats seulement si la simulation a été lancée
if (
    st.session_state.simulation_results is not None
    and st.session_state.simulation_results["time_stamps"]
):
    st.subheader("Résultats de la simulation")

    # Récupérer les résultats depuis session state
    stats = st.session_state.simulation_results

    # Afficher les statistiques principales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total livré", f"{stats['total_asphalt_delivered']:.1f} t")
        st.metric("Total posé", f"{stats['total_asphalt_laid']:.1f} t")
    with col2:
        st.metric("Utilisation poste d'enrobé", f"{stats['plant_utilization']:.1f}%")
        st.metric("Utilisation atelier", f"{stats['machine_utilization']:.1f}%")
    with col3:
        duration_hours = stats["time_stamps"][-1] / 60
        st.metric("Durée simulation", f"{duration_hours:.2f} h")
        st.metric("Utilisation camions", f"{stats['truck_utilization']:.1f}%")

    # Graphique 1 : évolution des camions
    data_dict = {
        "time_stamps": stats["time_stamps"],
        "trucks_at_plant": stats["trucks_at_plant"],
        "trucks_at_machine": stats["trucks_at_machine"],
        "trucks_on_road": stats["trucks_on_road"],
    }
    df_line = pd.DataFrame(data_dict)
    st.subheader("Graphique du mouvement des camions")
    st.line_chart(
        df_line.set_index("time_stamps"), use_container_width=True, height=500
    )

    # Graphique 2 : taux d'utilisation des camions
    metrics = ["Poste d'enrobé", "Atelier", "Camions"]
    utilizations = [
        round(stats["plant_utilization"], 2),
        round(stats["machine_utilization"], 2),
        round(stats["truck_utilization"], 2),
    ]
    df_bar = pd.DataFrame({"Metrics": metrics, "Utilizations": utilizations})
    st.subheader("Efficience par rapport à la durée totale")
    st.bar_chart(
        df_bar.set_index("Metrics"),
        use_container_width=True,
        height=500,
    )

else:
    st.info("Cliquez sur 'Lancer la simulation' pour voir les résultats.")


st.divider()
st.header("2. Impact du nombre de camions sur la productivité")
max_camion = st.number_input(
    "Nombre max des camions pour la simulation", min_value=1, value=n_camion + 5
)
sim_distance = st.number_input(
    "Distance pour la simulation (mètres)",
    min_value=500,
    value=distance,
)
sim_q_cible = st.number_input(
    "Quantité cible d'enrobé pour la simulation (tonnes)", min_value=40, value=q_cible
)


def optimize_truck_fleet(distance, target_quantity, max_trucks=max_camion):
    """
    Optimiser le nombre de camions pour maximiser l'utilisation de l'atelier.

    Arguments :
        distance: Distance entre le poste d'enrobé et l'atelier (mètres)
        target_quantity: Quantité cible d'enrobé à produire/poser (tonnes)
        max_trucks: Nombre maximum de camions à tester

    Retourne :
        Dictionnaire avec les résultats d'optimisation et le nombre optimal de camions
    """
    results = []

    for num_trucks in range(1, max_trucks + 1):
        sim = AsphaltSimulation(num_trucks, distance, target_quantity)
        sim.run_simulation()
        results.append(
            {
                "trucks": num_trucks,
                "plant_utilization": sim.stats["plant_utilization"],
                "machine_utilization": sim.stats["machine_utilization"],
                "truck_utilization": sim.stats["truck_utilization"],
                "total_asphalt_laid": sim.stats["total_asphalt_laid"],
                "avg_at_plant": np.mean(sim.stats["trucks_at_plant"]),
                "avg_at_machine": np.mean(sim.stats["trucks_at_machine"]),
                "avg_on_road": np.mean(sim.stats["trucks_on_road"]),
            }
        )
    # Trouver le nombre de camions qui maximise l'utilisation de l'atelier
    machine_utils = [r["machine_utilization"] for r in results]
    if all(mu == 0 for mu in machine_utils):
        optimal_trucks = None
        max_machine_utilization = 0
    else:
        optimal_idx = int(np.argmax(machine_utils))
        optimal_trucks = results[optimal_idx]["trucks"]
        max_machine_utilization = results[optimal_idx]["machine_utilization"]
    return {
        "results": results,
        "optimal_trucks": optimal_trucks,
        "max_machine_utilization": max_machine_utilization,
    }


optimization_results = optimize_truck_fleet(
    distance=sim_distance, target_quantity=sim_q_cible, max_trucks=max_camion
)
df_optimisation = pd.DataFrame(optimization_results["results"])
# Rename columns to French for better display
df_optimisation = df_optimisation.rename(
    columns={
        "trucks": "Camions",
        "plant_utilization": "Utilisation Poste (%)",
        "machine_utilization": "Utilisation Atelier (%)",
        "truck_utilization": "Utilisation Camions (%)",
        "total_asphalt_laid": "Total Posé (t)",
        "avg_at_plant": "Moy. au Poste",
        "avg_at_machine": "Moy. à l'Atelier",
        "avg_on_road": "Moy. sur Route",
    }
)
if st.button("Afficher les résultats d'optimisation"):
    st.subheader("Tableau des résultats d'optimisation")
    st.dataframe(df_optimisation)

    st.subheader("Evolution de l'efficience en fonction du nombre de camions")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Utilisation Atelier (%)"],
            name="Util.Atelier",  # Style name/legend entry with html tags
            connectgaps=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Utilisation Poste (%)"],
            name="Util.Poste",  # Style name/legend entry with html tags
            connectgaps=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Utilisation Camions (%)"],
            name="Util.Camions",  # Style name/legend entry with html tags
            connectgaps=True,
        )
    )
    st.plotly_chart(fig, theme="streamlit")

    st.subheader(
        "Evolution de la moyenne à chaque station en fonction du nombre de camions"
    )
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Moy. au Poste"],
            name="Moy. au Poste",  # Style name/legend entry with html tags
            connectgaps=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Moy. à l'Atelier"],
            name="Moy. à l'Atelier",  # Style name/legend entry with html tags
            connectgaps=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Moy. sur Route"],
            name="Moy. sur Route",  # Style name/legend entry with html tags
            connectgaps=True,
        )
    )
    st.plotly_chart(fig, theme="streamlit")

    def find_stabilization_point(df, column_name, threshold=2):
        """
        Analyzes a DataFrame column to find the first occurrence where the
        difference between consecutive values is less than a given threshold.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            column_name (str): The name of the column to analyze.
            threshold (float): The stabilization threshold. The function will find
                            the first point where the absolute difference between
                            consecutive values is less than this value.
                            Defaults to 0.02 (2%).

        Returns:
            dict: A dictionary containing the index, the two values, and their
                difference if a stabilization point is found.
            None: If no such point is found in the data.
        """
        # Check if the column exists in the DataFrame
        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in the DataFrame.")
            return None

        # Iterate through the DataFrame to find the stabilization point
        for i in range(len(df) - 1):
            # Calculate the absolute difference between the current and next value
            current_value = df[column_name].iloc[i]
            next_value = df[column_name].iloc[i + 1]
            diff = next_value - current_value

            # Check if the absolute difference is less than the threshold
            if abs(diff) < threshold:
                # If it is, we've found our stabilization point
                result = {
                    "index": i,
                    "stabilization_camions": df["Camions"].iloc[i],
                    "value_at_index": current_value,
                    "next_value": next_value,
                    "difference": diff,
                }
                return result

        # If the loop completes without finding a stabilization point
        return None

    # Find the stabilization point for the "Utilisation Atelier (%)" column
    stabilization_point = find_stabilization_point(
        df_optimisation, "Utilisation Atelier (%)"
    )
    if stabilization_point:
        st.subheader("Nombre de camions optimal")
        st.write(
            f"Le nombre de camions optimal est atteint avec {stabilization_point['stabilization_camions']} camions."
        )
        st.write(
            f"Valeur à l'index {stabilization_point['index']}: {stabilization_point['value_at_index']:.2f}%"
        )
        st.write(
            f"Valeur suivante: {stabilization_point['next_value']:.2f}%, "
            f"Différence: {stabilization_point['difference']:.2f}%"
        )
