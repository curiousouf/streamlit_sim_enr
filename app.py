import streamlit as st
import simpy
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Simulateur de Rotation des Camions",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling with dark mode compatibility
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #5D6D7E 0%, #85929E 100%);
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white !important;
    }
    .main-header h1 {
        font-size: 2.2rem !important;
        margin-bottom: 0.5rem !important;
    }
    .main-header p {
        font-size: 1rem !important;
        margin-bottom: 0 !important;
    }
    .section-header {
        background: linear-gradient(90deg, #7B8A8B 0%, #A2A9AF 100%);
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.8rem 0;
        color: white !important;
        text-align: center;
    }
    .section-header h2 {
        font-size: 1.5rem !important;
        margin-bottom: 0 !important;
    }
    .metric-card {
        background: var(--background-color, white);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #7B8A8B;
    }
    .stButton > button {
        background: linear-gradient(90deg, #7B8A8B 0%, #A2A9AF 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.5rem 2rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    .data-section {
        background: rgba(248, 249, 250, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
    }
    .data-section h3 {
        font-size: 1.1rem !important;
        margin-bottom: 0.8rem !important;
        color: #5D6D7E !important;
    }
    
    /* Dark mode specific styles */
    [data-theme="dark"] .data-section {
        background: rgba(26, 32, 44, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-theme="dark"] .data-section h3 {
        color: #A2A9AF !important;
    }
    
    /* Info boxes with dark mode support */
    .info-box {
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.8rem 0;
        border-left: 4px solid;
    }
    
    .info-box-success {
        background: rgba(232, 245, 233, 0.8);
        border-left-color: #6C7A6B;
        color: #2E3B2E;
    }
    
    .info-box-warning {
        background: rgba(255, 248, 225, 0.8);
        border-left-color: #B8860B;
        color: #7D5A00;
    }
    
    .info-box-info {
        background: rgba(227, 242, 253, 0.8);
        border-left-color: #5A7A95;
        color: #1E3A52;
    }
    
    .info-box-error {
        background: rgba(248, 235, 234, 0.8);
        border-left-color: #A85A5A;
        color: #6B2C2C;
    }
    
    /* Dark mode overrides for info boxes */
    [data-theme="dark"] .info-box-success {
        background: rgba(108, 122, 107, 0.2);
        color: #B8D4B8;
    }
    
    [data-theme="dark"] .info-box-warning {
        background: rgba(184, 134, 11, 0.2);
        color: #E6CC80;
    }
    
    [data-theme="dark"] .info-box-info {
        background: rgba(90, 122, 149, 0.2);
        color: #A8C5E0;
    }
    
    [data-theme="dark"] .info-box-error {
        background: rgba(168, 90, 90, 0.2);
        color: #E0A8A8;
    }
    
    /* Text color adjustments */
    .info-box h3 {
        color: inherit !important;
        margin: 0 0 0.4rem 0 !important;
        font-size: 1.1rem !important;
    }
    
    .info-box p {
        color: inherit !important;
        margin: 0 !important;
        font-size: 0.9rem !important;
    }
    
    /* Ensure proper contrast in all themes */
    .main-header h1,
    .main-header p,
    .section-header h2 {
        color: white !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Main title with enhanced styling
st.markdown(
    """
<div class="main-header">
    <h1>🚛 Simulateur de Rotation des Camions</h1>
    <p>Étude d'optimisation pour la mise en œuvre d'enrobé</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="section-header">
    <h2>📊 1. Simulation de la rotation des camions</h2>
</div>
""",
    unsafe_allow_html=True,
)


col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
    <div class="data-section">
        <h3>⚙️ Données de base</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    c_poste = st.number_input(
        "🏭 Débit du poste d'enrobé (tonnes/heure)",
        min_value=50,
        value=180,
        help="Capacité de production du poste d'enrobé",
    )
    c_camions = st.number_input(
        "🚛 Capacité des camions (tonnes)",
        min_value=5,
        value=40,
        help="Charge maximale que peut transporter un camion",
    )
    d_bache = st.number_input(
        "⏱️ Durée de bâchage (minutes)",
        min_value=1,
        value=8,
        help="Temps nécessaire pour bâcher le camion",
    )
    t_dechargement = st.number_input(
        "⏳ Temps de déchargement au finisher (minutes)",
        min_value=1,
        value=8,
        help="Temps pour décharger l'enrobé à l'atelier",
    )
    v_allez = st.slider(
        "🏃 Intervalle vitesse aller (km/h)",
        0,
        100,
        (15, 25),
        5,
        help="Vitesse des camions chargés vers l'atelier",
    )

with col2:
    st.markdown(
        """
    <div class="data-section">
        <h3>🎯 Données de production</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    n_camion = st.number_input(
        "🚚 Nombre de camions",
        min_value=1,
        value=10,
        help="Nombre total de camions dans la flotte",
    )
    distance = st.number_input(
        "📏 Distance poste d'enrobé ↔ atelier (mètres)",
        min_value=500,
        value=1000,
        help="Distance entre le poste d'enrobé et l'atelier",
    )
    q_cible = st.number_input(
        "🎯 Quantité cible d'enrobé (tonnes)",
        min_value=40,
        value=1000,
        help="Objectif de production total",
    )
    n_start = st.number_input(
        "🚥 Camions requis pour démarrer la M.O",
        min_value=1,
        value=1,
        help="Nombre de camions nécessaires à l'atelier avant de commencer la pose",
    )
    v_retour = st.slider(
        "🔄 Intervalle vitesse retour (km/h)",
        0,
        100,
        (30, 40),
        5,
        help="Vitesse des camions vides retournant au poste",
    )

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
            "machine_usage_duration": 0,
            "machine_idle_time": 0,
            "machine_longest_gap": 0,
        }

        # Suivi de la production du poste d'enrobé
        self.plant_production = 0
        self.plant_start_time = 0
        self.machine_laying_time = 0

        # Suivi des temps d'utilisation de l'atelier
        self.machine_first_usage_time = None
        self.machine_last_usage_time = None
        self.machine_usage_periods = []  # Liste des périodes d'utilisation (start_time, end_time)

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
                usage_start_time = self.env.now

                # Enregistrer la première utilisation de l'atelier
                if self.machine_first_usage_time is None:
                    self.machine_first_usage_time = usage_start_time

                yield self.env.timeout(self.UNLOADING_TIME)

                usage_end_time = self.env.now

                # Enregistrer cette période d'utilisation
                self.machine_usage_periods.append((usage_start_time, usage_end_time))

                # Mettre à jour la dernière utilisation de l'atelier
                self.machine_last_usage_time = usage_end_time
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

        # Calculer la durée d'utilisation de l'atelier (de la première à la dernière utilisation)
        if (
            self.machine_first_usage_time is not None
            and self.machine_last_usage_time is not None
        ):
            self.stats["machine_usage_duration"] = (
                self.machine_last_usage_time - self.machine_first_usage_time
            )

            # Calculer le temps d'inactivité et le plus grand gap
            if len(self.machine_usage_periods) > 1:
                # Trier les périodes par heure de début (au cas où)
                sorted_periods = sorted(self.machine_usage_periods, key=lambda x: x[0])

                total_idle_time = 0
                longest_gap = 0

                # Calculer les gaps entre les utilisations consécutives
                for i in range(len(sorted_periods) - 1):
                    current_end = sorted_periods[i][1]
                    next_start = sorted_periods[i + 1][0]
                    gap = next_start - current_end

                    total_idle_time += gap
                    longest_gap = max(longest_gap, gap)

                self.stats["machine_idle_time"] = total_idle_time
                self.stats["machine_longest_gap"] = longest_gap
            else:
                self.stats["machine_idle_time"] = 0
                self.stats["machine_longest_gap"] = 0
        else:
            self.stats["machine_usage_duration"] = 0
            self.stats["machine_idle_time"] = 0
            self.stats["machine_longest_gap"] = 0


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


# Action buttons with enhanced styling
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

with col_btn1:
    if st.button("🚀 Lancer la simulation", use_container_width=True):
        with st.spinner("⏳ Simulation en cours..."):
            run_sim()

with col_btn2:
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.simulation_results = None
        st.success("✅ Résultats réinitialisés!")

with col_btn3:
    st.info(
        "💡 Ajustez les paramètres ci-dessus puis lancez la simulation pour voir les résultats"
    )

# Afficher les résultats seulement si la simulation a été lancée
if (
    st.session_state.simulation_results is not None
    and st.session_state.simulation_results["time_stamps"]
):
    st.markdown(
        """
    <div class="section-header">
        <h2>📈 Résultats de la simulation</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Récupérer les résultats depuis session state
    stats = st.session_state.simulation_results

    # Afficher les statistiques principales avec des icônes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "📦 Total livré",
            f"{stats['total_asphalt_delivered']:.1f} t",
            help="Quantité totale d'enrobé livré",
        )
        st.metric(
            "🏭 Utilisation poste",
            f"{stats['plant_utilization']:.1f}%",
            help="Pourcentage d'utilisation du poste d'enrobé",
        )
    with col2:
        st.metric(
            "🛣️ Total posé",
            f"{stats['total_asphalt_laid']:.1f} t",
            help="Quantité totale d'enrobé posé",
        )
        st.metric(
            "⚙️ Utilisation atelier",
            f"{stats['machine_utilization']:.1f}%",
            help="Pourcentage d'utilisation de l'atelier",
        )
    with col3:
        duration_hours = stats["time_stamps"][-1] / 60
        st.metric(
            "⏰ Durée simulation",
            f"{duration_hours:.2f} h",
            help="Durée totale de la simulation",
        )
        st.metric(
            "🚛 Utilisation camions",
            f"{stats['truck_utilization']:.1f}%",
            help="Pourcentage d'utilisation des camions",
        )
    with col4:
        machine_usage_hours = stats.get("machine_usage_duration", 0) / 60
        st.metric(
            "🕐 Durée usage atelier",
            f"{machine_usage_hours:.2f} h",
            help="Durée d'utilisation effective de l'atelier",
        )
        if duration_hours > 0:
            machine_coverage = (machine_usage_hours / duration_hours) * 100
            st.metric(
                "📊 Couverture atelier",
                f"{machine_coverage:.1f}%",
                help="Pourcentage de couverture de l'atelier",
            )

    # Afficher les statistiques d'inactivité de l'atelier
    st.markdown(
        """
    <div class="info-box info-box-warning">
        <h3>⏸️ Analyse des périodes d'inactivité de l'atelier</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1_idle, col2_idle, col3_idle = st.columns(3)
    with col1_idle:
        machine_idle_hours = stats.get("machine_idle_time", 0) / 60
        st.metric(
            "⏸️ Temps total inactif",
            f"{machine_idle_hours:.2f} h",
            help="Temps total d'inactivité de l'atelier",
        )
    with col2_idle:
        machine_longest_gap_hours = stats.get("machine_longest_gap", 0) / 60
        st.metric(
            "⏱️ Plus long arrêt",
            f"{machine_longest_gap_hours*60:.2f} min",
            help="Durée du plus long arrêt de l'atelier",
        )
    with col3_idle:
        if machine_usage_hours > 0:
            idle_percentage = (machine_idle_hours / machine_usage_hours) * 100
            st.metric(
                "📉 % inactivité/usage",
                f"{idle_percentage:.1f}%",
                help="Pourcentage d'inactivité par rapport au temps d'usage",
            )

    # Graphique 1 : évolution des camions
    data_dict = {
        "time_stamps": stats["time_stamps"],
        "trucks_at_plant": stats["trucks_at_plant"],
        "trucks_at_machine": stats["trucks_at_machine"],
        "trucks_on_road": stats["trucks_on_road"],
    }
    df_line = pd.DataFrame(data_dict)

    st.markdown(
        """
    <div class="info-box info-box-success">
        <h3>🔄 Graphique de la rotation des camions</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.line_chart(
        df_line.set_index("time_stamps"), use_container_width=True, height=500
    )

    # Graphique 2 : taux d'utilisation des camions
    metrics = ["🏭 Poste d'enrobé", "⚙️ Atelier", "🚛 Camions"]
    utilizations = [
        round(stats["plant_utilization"], 2),
        round(stats["machine_utilization"], 2),
        round(stats["truck_utilization"], 2),
    ]
    df_bar = pd.DataFrame({"Metrics": metrics, "Utilizations": utilizations})

    st.markdown(
        """
    <div class="info-box info-box-info">
        <h3>📊 Efficience par rapport à la durée totale</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.bar_chart(
        df_bar.set_index("Metrics"),
        use_container_width=True,
        height=500,
    )

else:
    st.markdown(
        """
    <div class="info-box info-box-error">
        <h3>🎯 Prêt à simuler ?</h3>
        <p>Cliquez sur '<strong>🚀 Lancer la simulation</strong>' pour voir les résultats détaillés</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
<div class="section-header">
    <h2>🎯 2. Impact du nombre de camions sur la productivité</h2>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="data-section">
    <h3>⚙️ Paramètres d'optimisation</h3>
</div>
""",
    unsafe_allow_html=True,
)

col_opt1, col_opt2, col_opt3 = st.columns(3)

with col_opt1:
    max_camion = st.number_input(
        "🔢 Nombre max de camions à tester",
        min_value=1,
        value=n_camion + 5,
        help="Limite supérieure pour l'analyse d'optimisation",
    )

with col_opt2:
    sim_distance = st.number_input(
        "📏 Distance pour la simulation (mètres)",
        min_value=500,
        value=distance,
        help="Distance utilisée pour l'analyse d'optimisation",
    )

with col_opt3:
    sim_q_cible = st.number_input(
        "🎯 Quantité cible pour la simulation (tonnes)",
        min_value=40,
        value=q_cible,
        help="Objectif de production pour l'analyse",
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
                "machine_idle_time": sim.stats["machine_idle_time"],
                "machine_longest_gap": sim.stats["machine_longest_gap"],
                "machine_usage_duration": sim.stats["machine_usage_duration"],
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
        "machine_idle_time": "Temps Inactif Atelier (min)",
        "machine_longest_gap": "Plus Long Arrêt (min)",
        "machine_usage_duration": "Durée Usage Atelier (min)",
    }
)
if st.button("📊 Afficher les résultats d'optimisation", use_container_width=True):
    with st.spinner("🔄 Analyse d'optimisation en cours..."):
        # Progress bar simulation
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)

        st.markdown(
            """
        <div class="info-box info-box-success">
            <h3>📋 Tableau des résultats d'optimisation</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Style the dataframe
        styled_df = df_optimisation.style.format(
            {
                "Utilisation Poste (%)": "{:.1f}%",
                "Utilisation Atelier (%)": "{:.1f}%",
                "Utilisation Camions (%)": "{:.1f}%",
                "Total Posé (t)": "{:.0f}",
                "Moy. au Poste": "{:.1f}",
                "Moy. à l'Atelier": "{:.1f}",
                "Moy. sur Route": "{:.1f}",
                "Temps Inactif Atelier (min)": "{:.1f}",
                "Plus Long Arrêt (min)": "{:.1f}",
                "Durée Usage Atelier (min)": "{:.1f}",
            }
        ).background_gradient(subset=["Utilisation Atelier (%)"], cmap="RdYlGn")

        st.dataframe(styled_df, use_container_width=True)

    st.markdown(
        """
    <div class="info-box info-box-success">
        <h3>📈 Evolution de l'efficience en fonction du nombre de camions</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Utilisation Atelier (%)"],
            name="⚙️ Util.Atelier",
            line=dict(color="#FF6B35", width=3),
            connectgaps=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Utilisation Poste (%)"],
            name="🏭 Util.Poste",
            line=dict(color="#4CAF50", width=3),
            connectgaps=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Utilisation Camions (%)"],
            name="🚛 Util.Camions",
            line=dict(color="#2196F3", width=3),
            connectgaps=True,
        )
    )

    fig.update_layout(
        title="Évolution de l'efficience",
        xaxis_title="Nombre de Camions",
        yaxis_title="Utilisation (%)",
        template="plotly_white",
        showlegend=True,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
    <div class="info-box info-box-info">
        <h3>🚛 Distribution des camions par station</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Moy. au Poste"],
            name="🏭 Moy. au Poste",
            line=dict(color="#FF9800", width=3),
            connectgaps=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Moy. à l'Atelier"],
            name="⚙️ Moy. à l'Atelier",
            line=dict(color="#9C27B0", width=3),
            connectgaps=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Moy. sur Route"],
            name="🛣️ Moy. sur Route",
            line=dict(color="#607D8B", width=3),
            connectgaps=True,
        )
    )

    fig.update_layout(
        title="Distribution moyenne des camions par station",
        xaxis_title="Nombre total de Camions",
        yaxis_title="Nombre moyen de camions",
        template="plotly_white",
        showlegend=True,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
    <div class="info-box info-box-warning">
        <h3>⏸️ Analyse des temps d'inactivité</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Temps Inactif Atelier (min)"],
            name="⏸️ Temps Total Inactif",
            line=dict(color="#F44336", width=3),
            connectgaps=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_optimisation["Camions"],
            y=df_optimisation["Plus Long Arrêt (min)"],
            name="⏱️ Plus Long Arrêt",
            line=dict(color="#FF5722", width=3),
            connectgaps=True,
        )
    )

    fig.update_layout(
        title="Évolution des temps d'inactivité de l'atelier",
        xaxis_title="Nombre de Camions",
        yaxis_title="Temps (minutes)",
        template="plotly_white",
        showlegend=True,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

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
        st.markdown(
            """
        <div class="info-box info-box-info">
            <h3>🎯 Nombre de camions optimal</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Create columns for better layout
        col_opt_1, col_opt_2 = st.columns(2)

        with col_opt_1:
            st.success(
                f"🚛 **Optimal:** {stabilization_point['stabilization_camions']} camions"
            )
            st.info(
                f"📊 **Utilisation à l'optimum:** {stabilization_point['value_at_index']:.2f}%"
            )

        with col_opt_2:
            st.info(f"📈 **Valeur suivante:** {stabilization_point['next_value']:.2f}%")
            st.warning(
                f"📉 **Gain marginal:** {stabilization_point['difference']:.2f}%"
            )

        st.markdown(
            """
        <div class="info-box info-box-success">
            <p><strong>💡 Interprétation:</strong> Au-delà de ce nombre optimal, 
            l'ajout de camions supplémentaires n'améliore plus significativement 
            l'utilisation de l'atelier (gain < 2%).</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class="info-box info-box-error">
            <p><strong>⚠️ Attention:</strong> Aucun point de stabilisation détecté dans la plage testée. 
            Essayez d'augmenter le nombre maximum de camions pour l'analyse.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
