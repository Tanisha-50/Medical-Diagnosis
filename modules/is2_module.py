"""
Intelligent Systems II Module for Medical Diagnosis
Fuzzy Logic + Genetic Algorithm for treatment recommendation
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from deap import base, creator, tools, algorithms
import random
from modules.logger import log_to_sidebar, log_to_main

class IntelligentSystemsModule:
    """Intelligent Systems II module for treatment recommendation using Fuzzy Logic + GA"""
    
    def __init__(self):
        self.fuzzy_system = None
        self.treatment_options = []
        self.ga_results = None
        self.is_initialized = False
        
        # Treatment options database
        self.treatment_database = {
            'Pneumonia': [
                {'name': 'Antibiotics (Amoxicillin)', 'dosage': '500mg', 'frequency': '3x daily', 'duration': '7-10 days'},
                {'name': 'Oxygen Therapy', 'dosage': '2-4 L/min', 'frequency': 'Continuous', 'duration': 'As needed'},
                {'name': 'Chest Physiotherapy', 'dosage': 'N/A', 'frequency': '2x daily', 'duration': '5-7 days'}
            ],
            'COVID-19': [
                {'name': 'Antiviral (Remdesivir)', 'dosage': '200mg', 'frequency': '1x daily', 'duration': '5 days'},
                {'name': 'Dexamethasone', 'dosage': '6mg', 'frequency': '1x daily', 'duration': '10 days'},
                {'name': 'Oxygen Support', 'dosage': '2-6 L/min', 'frequency': 'Continuous', 'duration': 'As needed'}
            ],
            'Tuberculosis': [
                {'name': 'Isoniazid', 'dosage': '300mg', 'frequency': '1x daily', 'duration': '6 months'},
                {'name': 'Rifampin', 'dosage': '600mg', 'frequency': '1x daily', 'duration': '6 months'},
                {'name': 'Ethambutol', 'dosage': '1200mg', 'frequency': '1x daily', 'duration': '2 months'}
            ],
            'Diabetes': [
                {'name': 'Metformin', 'dosage': '500mg', 'frequency': '2x daily', 'duration': 'Long-term'},
                {'name': 'Insulin Therapy', 'dosage': 'Variable', 'frequency': 'As needed', 'duration': 'Long-term'},
                {'name': 'Dietary Counseling', 'dosage': 'N/A', 'frequency': 'Monthly', 'duration': 'Ongoing'}
            ],
            'Hypertension': [
                {'name': 'ACE Inhibitor', 'dosage': '10mg', 'frequency': '1x daily', 'duration': 'Long-term'},
                {'name': 'Diuretic', 'dosage': '25mg', 'frequency': '1x daily', 'duration': 'Long-term'},
                {'name': 'Lifestyle Modification', 'dosage': 'N/A', 'frequency': 'Daily', 'duration': 'Ongoing'}
            ]
        }
    
    def run_module(self):
        """Main function to run the Intelligent Systems II module"""
        
        # Sidebar for GA parameters
        st.sidebar.markdown("### 🧬 Genetic Algorithm Parameters")
        population_size = st.sidebar.slider("Population Size", 10, 100, 50)
        generations = st.sidebar.slider("Generations", 10, 100, 30)
        mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.3, 0.1)
        crossover_rate = st.sidebar.slider("Crossover Rate", 0.5, 1.0, 0.8)
        
        # Risk tolerance
        risk_tolerance = st.sidebar.selectbox(
            "Risk Tolerance",
            ["Conservative", "Moderate", "Aggressive"],
            help="Higher risk tolerance allows more experimental treatments"
        )
        
        # Main content area
        tab1, tab2, tab3 = st.tabs(["🎯 Treatment Input", "🧠 Fuzzy Logic Analysis", "🧬 GA Optimization"])
        
        with tab1:
            self._treatment_input_tab()
        
        with tab2:
            self._fuzzy_logic_tab()
        
        with tab3:
            self._ga_optimization_tab(population_size, generations, mutation_rate, crossover_rate, risk_tolerance)
    
    def _treatment_input_tab(self):
        """Treatment input and disease selection tab"""
        st.markdown("### Treatment Recommendation Input")
        
        # Disease selection
        selected_disease = st.selectbox(
            "Select Predicted Disease",
            list(self.treatment_database.keys()),
            help="Select the disease from Deep Learning module prediction"
        )
        
        # Patient condition inputs
        st.markdown("#### Patient Condition Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Patient Age", min_value=0, max_value=120, value=50)
            weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=60, max_value=200, value=120)
        
        with col2:
            temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)
            oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=70, max_value=100, value=98)
        
        # Store patient data
        patient_data = {
            'disease': selected_disease,
            'age': age,
            'weight': weight,
            'blood_pressure': blood_pressure,
            'temperature': temperature,
            'heart_rate': heart_rate,
            'oxygen_saturation': oxygen_saturation
        }
        
        # Store in session state
        st.session_state.patient_data = patient_data
        
        # Display available treatments
        st.markdown("#### Available Treatment Options")
        if selected_disease in self.treatment_database:
            treatments = self.treatment_database[selected_disease]
            for i, treatment in enumerate(treatments, 1):
                with st.expander(f"Treatment {i}: {treatment['name']}"):
                    st.write(f"**Dosage:** {treatment['dosage']}")
                    st.write(f"**Frequency:** {treatment['frequency']}")
                    st.write(f"**Duration:** {treatment['duration']}")
        
        # Initialize fuzzy system
        if st.button("🔧 Initialize Fuzzy System", type="primary"):
            self._initialize_fuzzy_system()
            st.success("✅ Fuzzy system initialized!")
    
    def _fuzzy_logic_tab(self):
        """Fuzzy logic analysis tab"""
        st.markdown("### Fuzzy Logic Analysis")
        
        if not self.is_initialized:
            st.warning("⚠️ Please initialize the fuzzy system first.")
            return
        
        if 'patient_data' not in st.session_state:
            st.warning("⚠️ Please provide patient data first.")
            return
        
        patient_data = st.session_state.patient_data
        
        # Perform fuzzy analysis
        severity_score = self._perform_fuzzy_analysis(patient_data)
        
        # Display results
        st.markdown("#### Fuzzy Logic Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Severity Score", f"{severity_score:.2f}")
        
        with col2:
            if severity_score < 0.3:
                severity_level = "Mild"
                color = "green"
            elif severity_score < 0.7:
                severity_level = "Moderate"
                color = "orange"
            else:
                severity_level = "Severe"
                color = "red"
            
            st.markdown(f"**Severity Level:** <span style='color: {color}'>{severity_level}</span>", unsafe_allow_html=True)
        
        with col3:
            risk_level = self._calculate_risk_level(severity_score, patient_data)
            st.metric("Risk Level", risk_level)
        
        # Display fuzzy membership functions
        self._display_fuzzy_membership_functions()
        
        # Store severity score for GA
        st.session_state.severity_score = severity_score
    
    def _ga_optimization_tab(self, population_size, generations, mutation_rate, crossover_rate, risk_tolerance):
        """Genetic Algorithm optimization tab"""
        st.markdown("### Genetic Algorithm Optimization")
        
        if 'severity_score' not in st.session_state:
            st.warning("⚠️ Please complete fuzzy logic analysis first.")
            return
        
        if st.button("🧬 Run Genetic Algorithm", type="primary"):
            self._run_genetic_algorithm(population_size, generations, mutation_rate, crossover_rate, risk_tolerance)
        
        if self.ga_results:
            self._display_ga_results()
    
    def _initialize_fuzzy_system(self):
        """Initialize the fuzzy logic system"""
        log_to_sidebar("Initializing fuzzy logic system...")
        
        # Create fuzzy variables
        temperature = ctrl.Antecedent(np.arange(35, 42, 0.1), 'temperature')
        blood_pressure = ctrl.Antecedent(np.arange(60, 200, 1), 'blood_pressure')
        heart_rate = ctrl.Antecedent(np.arange(40, 200, 1), 'heart_rate')
        oxygen_saturation = ctrl.Antecedent(np.arange(70, 100, 0.1), 'oxygen_saturation')
        age = ctrl.Antecedent(np.arange(0, 120, 1), 'age')
        
        severity = ctrl.Consequent(np.arange(0, 1, 0.01), 'severity')
        
        # Define membership functions for temperature
        temperature['low'] = fuzz.trimf(temperature.universe, [35, 35, 36.5])
        temperature['normal'] = fuzz.trimf(temperature.universe, [36, 37.5, 38.5])
        temperature['high'] = fuzz.trimf(temperature.universe, [37.5, 39, 42])
        temperature['very_high'] = fuzz.trimf(temperature.universe, [38.5, 40, 42])
        
        # Define membership functions for blood pressure
        blood_pressure['low'] = fuzz.trimf(blood_pressure.universe, [60, 60, 90])
        blood_pressure['normal'] = fuzz.trimf(blood_pressure.universe, [80, 110, 130])
        blood_pressure['high'] = fuzz.trimf(blood_pressure.universe, [120, 140, 160])
        blood_pressure['very_high'] = fuzz.trimf(blood_pressure.universe, [150, 180, 200])
        
        # Define membership functions for heart rate
        heart_rate['low'] = fuzz.trimf(heart_rate.universe, [40, 40, 60])
        heart_rate['normal'] = fuzz.trimf(heart_rate.universe, [50, 70, 90])
        heart_rate['high'] = fuzz.trimf(heart_rate.universe, [80, 100, 120])
        heart_rate['very_high'] = fuzz.trimf(heart_rate.universe, [110, 140, 200])
        
        # Define membership functions for oxygen saturation
        oxygen_saturation['low'] = fuzz.trimf(oxygen_saturation.universe, [70, 70, 85])
        oxygen_saturation['normal'] = fuzz.trimf(oxygen_saturation.universe, [80, 95, 100])
        oxygen_saturation['high'] = fuzz.trimf(oxygen_saturation.universe, [95, 100, 100])
        
        # Define membership functions for age
        age['young'] = fuzz.trimf(age.universe, [0, 0, 30])
        age['adult'] = fuzz.trimf(age.universe, [20, 50, 70])
        age['elderly'] = fuzz.trimf(age.universe, [60, 90, 120])
        
        # Define membership functions for severity
        severity['mild'] = fuzz.trimf(severity.universe, [0, 0, 0.3])
        severity['moderate'] = fuzz.trimf(severity.universe, [0.2, 0.5, 0.8])
        severity['severe'] = fuzz.trimf(severity.universe, [0.7, 1, 1])
        
        # Define rules
        rules = [
            # Temperature rules
            ctrl.Rule(temperature['very_high'], severity['severe']),
            ctrl.Rule(temperature['high'], severity['moderate']),
            ctrl.Rule(temperature['normal'], severity['mild']),
            ctrl.Rule(temperature['low'], severity['moderate']),
            
            # Blood pressure rules
            ctrl.Rule(blood_pressure['very_high'], severity['severe']),
            ctrl.Rule(blood_pressure['high'], severity['moderate']),
            ctrl.Rule(blood_pressure['normal'], severity['mild']),
            ctrl.Rule(blood_pressure['low'], severity['moderate']),
            
            # Heart rate rules
            ctrl.Rule(heart_rate['very_high'], severity['severe']),
            ctrl.Rule(heart_rate['high'], severity['moderate']),
            ctrl.Rule(heart_rate['normal'], severity['mild']),
            ctrl.Rule(heart_rate['low'], severity['moderate']),
            
            # Oxygen saturation rules
            ctrl.Rule(oxygen_saturation['low'], severity['severe']),
            ctrl.Rule(oxygen_saturation['normal'], severity['mild']),
            ctrl.Rule(oxygen_saturation['high'], severity['mild']),
            
            # Age rules
            ctrl.Rule(age['elderly'], severity['moderate']),
            ctrl.Rule(age['adult'], severity['mild']),
            ctrl.Rule(age['young'], severity['mild']),
        ]
        
        # Create control system
        self.fuzzy_system = ctrl.ControlSystem(rules)
        self.fuzzy_controller = ctrl.ControlSystemSimulation(self.fuzzy_system)
        
        self.is_initialized = True
        log_to_sidebar("Fuzzy logic system initialized", "SUCCESS")
    
    def _perform_fuzzy_analysis(self, patient_data):
        """Perform fuzzy logic analysis on patient data"""
        if not self.is_initialized:
            return 0.5  # Default value
        
        # Set input values
        self.fuzzy_controller.input['temperature'] = patient_data['temperature']
        self.fuzzy_controller.input['blood_pressure'] = patient_data['blood_pressure']
        self.fuzzy_controller.input['heart_rate'] = patient_data['heart_rate']
        self.fuzzy_controller.input['oxygen_saturation'] = patient_data['oxygen_saturation']
        self.fuzzy_controller.input['age'] = patient_data['age']
        
        # Compute
        self.fuzzy_controller.compute()
        
        # Get severity score
        severity_score = self.fuzzy_controller.output['severity']
        
        return severity_score
    
    def _calculate_risk_level(self, severity_score, patient_data):
        """Calculate risk level based on severity and patient data"""
        risk_factors = 0
        
        # Age risk
        if patient_data['age'] > 65:
            risk_factors += 1
        
        # Temperature risk
        if patient_data['temperature'] > 38.5:
            risk_factors += 1
        
        # Blood pressure risk
        if patient_data['blood_pressure'] > 140 or patient_data['blood_pressure'] < 90:
            risk_factors += 1
        
        # Heart rate risk
        if patient_data['heart_rate'] > 100 or patient_data['heart_rate'] < 60:
            risk_factors += 1
        
        # Oxygen saturation risk
        if patient_data['oxygen_saturation'] < 95:
            risk_factors += 1
        
        # Calculate risk level
        if risk_factors <= 1:
            return "Low"
        elif risk_factors <= 3:
            return "Medium"
        else:
            return "High"
    
    def _display_fuzzy_membership_functions(self):
        """Display fuzzy membership functions"""
        st.markdown("#### Fuzzy Membership Functions")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Temperature', 'Blood Pressure', 'Heart Rate', 'Oxygen Saturation', 'Age', 'Severity')
        )
        
        # Temperature membership functions
        temp_range = np.arange(35, 42, 0.1)
        fig.add_trace(go.Scatter(x=temp_range, y=fuzz.trimf(temp_range, [35, 35, 36.5]), name='Low', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=temp_range, y=fuzz.trimf(temp_range, [36, 37.5, 38.5]), name='Normal', line=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=temp_range, y=fuzz.trimf(temp_range, [37.5, 39, 42]), name='High', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=temp_range, y=fuzz.trimf(temp_range, [38.5, 40, 42]), name='Very High', line=dict(color='red')), row=1, col=1)
        
        # Blood pressure membership functions
        bp_range = np.arange(60, 200, 1)
        fig.add_trace(go.Scatter(x=bp_range, y=fuzz.trimf(bp_range, [60, 60, 90]), name='Low', line=dict(color='blue'), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=bp_range, y=fuzz.trimf(bp_range, [80, 110, 130]), name='Normal', line=dict(color='green'), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=bp_range, y=fuzz.trimf(bp_range, [120, 140, 160]), name='High', line=dict(color='orange'), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=bp_range, y=fuzz.trimf(bp_range, [150, 180, 200]), name='Very High', line=dict(color='red'), showlegend=False), row=1, col=2)
        
        # Heart rate membership functions
        hr_range = np.arange(40, 200, 1)
        fig.add_trace(go.Scatter(x=hr_range, y=fuzz.trimf(hr_range, [40, 40, 60]), name='Low', line=dict(color='blue'), showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=hr_range, y=fuzz.trimf(hr_range, [50, 70, 90]), name='Normal', line=dict(color='green'), showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=hr_range, y=fuzz.trimf(hr_range, [80, 100, 120]), name='High', line=dict(color='orange'), showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=hr_range, y=fuzz.trimf(hr_range, [110, 140, 200]), name='Very High', line=dict(color='red'), showlegend=False), row=1, col=3)
        
        # Oxygen saturation membership functions
        o2_range = np.arange(70, 100, 0.1)
        fig.add_trace(go.Scatter(x=o2_range, y=fuzz.trimf(o2_range, [70, 70, 85]), name='Low', line=dict(color='blue'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=o2_range, y=fuzz.trimf(o2_range, [80, 95, 100]), name='Normal', line=dict(color='green'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=o2_range, y=fuzz.trimf(o2_range, [95, 100, 100]), name='High', line=dict(color='green'), showlegend=False), row=2, col=1)
        
        # Age membership functions
        age_range = np.arange(0, 120, 1)
        fig.add_trace(go.Scatter(x=age_range, y=fuzz.trimf(age_range, [0, 0, 30]), name='Young', line=dict(color='blue'), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=age_range, y=fuzz.trimf(age_range, [20, 50, 70]), name='Adult', line=dict(color='green'), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=age_range, y=fuzz.trimf(age_range, [60, 90, 120]), name='Elderly', line=dict(color='orange'), showlegend=False), row=2, col=2)
        
        # Severity membership functions
        severity_range = np.arange(0, 1, 0.01)
        fig.add_trace(go.Scatter(x=severity_range, y=fuzz.trimf(severity_range, [0, 0, 0.3]), name='Mild', line=dict(color='green'), showlegend=False), row=2, col=3)
        fig.add_trace(go.Scatter(x=severity_range, y=fuzz.trimf(severity_range, [0.2, 0.5, 0.8]), name='Moderate', line=dict(color='orange'), showlegend=False), row=2, col=3)
        fig.add_trace(go.Scatter(x=severity_range, y=fuzz.trimf(severity_range, [0.7, 1, 1]), name='Severe', line=dict(color='red'), showlegend=False), row=2, col=3)
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def _run_genetic_algorithm(self, population_size, generations, mutation_rate, crossover_rate, risk_tolerance):
        """Run genetic algorithm for treatment optimization"""
        log_to_sidebar("Running genetic algorithm...")
        
        # Get patient data and severity
        patient_data = st.session_state.patient_data
        severity_score = st.session_state.severity_score
        
        # Setup GA
        self._setup_genetic_algorithm(population_size, mutation_rate, crossover_rate, risk_tolerance)
        
        # Run GA
        population, logbook = self._run_ga_optimization(generations)
        
        # Store results
        self.ga_results = {
            'population': population,
            'logbook': logbook,
            'best_individual': tools.selBest(population, 1)[0],
            'patient_data': patient_data,
            'severity_score': severity_score
        }
        
        log_to_sidebar("Genetic algorithm completed", "SUCCESS")
        st.success("✅ Genetic algorithm optimization completed!")
    
    def _setup_genetic_algorithm(self, population_size, mutation_rate, crossover_rate, risk_tolerance):
        """Setup genetic algorithm parameters"""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self.toolbox.register("attr_float", random.uniform, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=6)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register evaluation function
        self.toolbox.register("evaluate", self._evaluate_treatment)
        
        # Register genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Store parameters
        self.ga_params = {
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'risk_tolerance': risk_tolerance
        }
    
    def _evaluate_treatment(self, individual):
        """Evaluate treatment fitness"""
        # Extract treatment parameters
        dosage_intensity = individual[0]
        frequency = individual[1]
        duration = individual[2]
        combination_ratio = individual[3]
        monitoring_intensity = individual[4]
        risk_tolerance = individual[5]
        
        # Calculate fitness based on multiple factors
        effectiveness = self._calculate_effectiveness(dosage_intensity, frequency, duration)
        safety = self._calculate_safety(dosage_intensity, monitoring_intensity)
        efficiency = self._calculate_efficiency(frequency, duration)
        risk_score = self._calculate_risk_score(risk_tolerance, dosage_intensity)
        
        # Weighted fitness function
        fitness = (
            0.4 * effectiveness +
            0.3 * safety +
            0.2 * efficiency +
            0.1 * (1 - risk_score)  # Lower risk is better
        )
        
        return (fitness,)
    
    def _calculate_effectiveness(self, dosage_intensity, frequency, duration):
        """Calculate treatment effectiveness score"""
        # Higher dosage and frequency generally increase effectiveness
        # but with diminishing returns
        effectiveness = min(1.0, dosage_intensity * 0.5 + frequency * 0.3 + duration * 0.2)
        return effectiveness
    
    def _calculate_safety(self, dosage_intensity, monitoring_intensity):
        """Calculate treatment safety score"""
        # Higher monitoring and moderate dosage increase safety
        safety = min(1.0, monitoring_intensity * 0.6 + (1 - abs(dosage_intensity - 0.5)) * 0.4)
        return safety
    
    def _calculate_efficiency(self, frequency, duration):
        """Calculate treatment efficiency score"""
        # Lower frequency and duration increase efficiency
        efficiency = min(1.0, (1 - frequency) * 0.5 + (1 - duration) * 0.5)
        return efficiency
    
    def _calculate_risk_score(self, risk_tolerance, dosage_intensity):
        """Calculate risk score"""
        # Higher dosage increases risk, but risk tolerance can mitigate
        risk = max(0, dosage_intensity - risk_tolerance)
        return min(1.0, risk)
    
    def _run_ga_optimization(self, generations):
        """Run the genetic algorithm optimization"""
        # Create initial population
        population = self.toolbox.population(n=self.ga_params['population_size'])
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Logbook
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields
        
        # Evolution
        for gen in range(generations):
            # Select parents
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.ga_params['crossover_rate']:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < self.ga_params['mutation_rate']:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Record statistics
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            
            # Log progress
            if gen % 5 == 0:
                log_to_sidebar(f"Generation {gen}: Best fitness = {record['max']:.3f}")
        
        return population, logbook
    
    def _display_ga_results(self):
        """Display genetic algorithm results"""
        if not self.ga_results:
            return
        
        st.markdown("#### Genetic Algorithm Results")
        
        # Get best individual
        best_individual = self.ga_results['best_individual']
        logbook = self.ga_results['logbook']
        
        # Display best treatment parameters
        st.markdown("##### Optimal Treatment Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dosage Intensity", f"{best_individual[0]:.3f}")
            st.metric("Frequency", f"{best_individual[1]:.3f}")
        
        with col2:
            st.metric("Duration", f"{best_individual[2]:.3f}")
            st.metric("Combination Ratio", f"{best_individual[3]:.3f}")
        
        with col3:
            st.metric("Monitoring Intensity", f"{best_individual[4]:.3f}")
            st.metric("Risk Tolerance", f"{best_individual[5]:.3f}")
        
        # Display fitness evolution
        st.markdown("##### Fitness Evolution")
        
        gen = logbook.select("gen")
        avg_fitness = logbook.select("avg")
        max_fitness = logbook.select("max")
        min_fitness = logbook.select("min")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=gen, y=avg_fitness,
            mode='lines',
            name='Average Fitness',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=gen, y=max_fitness,
            mode='lines',
            name='Best Fitness',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=gen, y=min_fitness,
            mode='lines',
            name='Worst Fitness',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Genetic Algorithm Convergence",
            xaxis_title="Generation",
            yaxis_title="Fitness",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display treatment recommendations
        self._display_treatment_recommendations(best_individual)
    
    def _display_treatment_recommendations(self, best_individual):
        """Display treatment recommendations based on GA results"""
        st.markdown("##### Treatment Recommendations")
        
        patient_data = self.ga_results['patient_data']
        disease = patient_data['disease']
        
        if disease in self.treatment_database:
            treatments = self.treatment_database[disease]
            
            # Create recommendations table
            recommendations = []
            
            for i, treatment in enumerate(treatments):
                # Adjust treatment based on GA parameters
                adjusted_dosage = self._adjust_dosage(treatment['dosage'], best_individual[0])
                adjusted_frequency = self._adjust_frequency(treatment['frequency'], best_individual[1])
                adjusted_duration = self._adjust_duration(treatment['duration'], best_individual[2])
                
                # Calculate allocation and risk scores
                allocation = best_individual[3] * (i + 1) / len(treatments)
                risk_score = self._calculate_risk_score(best_individual[5], best_individual[0])
                effectiveness = self._calculate_effectiveness(best_individual[0], best_individual[1], best_individual[2])
                
                recommendations.append({
                    'Treatment Option': treatment['name'],
                    'Allocation': f"{allocation:.2f}",
                    'Risk Score': f"{risk_score:.3f}",
                    'Effectiveness': f"{effectiveness:.3f}",
                    'Adjusted Dosage': adjusted_dosage,
                    'Adjusted Frequency': adjusted_frequency,
                    'Adjusted Duration': adjusted_duration
                })
            
            # Display table
            recommendations_df = pd.DataFrame(recommendations)
            st.dataframe(recommendations_df, use_container_width=True)
            
            # Highlight best treatment
            best_treatment_idx = np.argmax([float(r['Effectiveness']) for r in recommendations])
            best_treatment = recommendations[best_treatment_idx]
            
            st.success(f"🎯 **Recommended Treatment:** {best_treatment['Treatment Option']} (Effectiveness: {best_treatment['Effectiveness']})")
    
    def _adjust_dosage(self, original_dosage, intensity_factor):
        """Adjust dosage based on intensity factor"""
        # Simple adjustment - in practice, this would be more sophisticated
        if 'mg' in original_dosage:
            try:
                value = float(original_dosage.replace('mg', ''))
                adjusted_value = value * (0.5 + intensity_factor)
                return f"{adjusted_value:.0f}mg"
            except:
                return original_dosage
        return original_dosage
    
    def _adjust_frequency(self, original_frequency, frequency_factor):
        """Adjust frequency based on frequency factor"""
        # Simple adjustment
        if 'daily' in original_frequency:
            try:
                value = float(original_frequency.replace('x daily', ''))
                adjusted_value = value * (0.5 + frequency_factor)
                return f"{adjusted_value:.1f}x daily"
            except:
                return original_frequency
        return original_frequency
    
    def _adjust_duration(self, original_duration, duration_factor):
        """Adjust duration based on duration factor"""
        # Simple adjustment
        if 'days' in original_duration:
            try:
                value = float(original_duration.replace(' days', ''))
                adjusted_value = value * (0.5 + duration_factor)
                return f"{adjusted_value:.0f} days"
            except:
                return original_duration
        return original_duration
