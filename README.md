# Simulation de la Logistique de Pose d'Enrobés

Une application Streamlit pour simuler et optimiser la logistique des opérations de pose d'enrobés dans les projets de construction routière.

## 📋 Description

Cette application simule le processus complet de transport et de pose d'enrobés, depuis le poste de production jusqu'à l'atelier de pose. Elle permet d'analyser l'efficacité du système logistique et d'optimiser le nombre de camions pour maximiser la productivité.

## 🎯 Fonctionnalités

### 1. Simulation du Mouvement des Camions

- **Simulation en temps réel** du cycle complet des camions
- **Paramètres configurables** :
  - Capacité du poste d'enrobé (tonnes/heure)
  - Capacité des camions (tonnes)
  - Durée de bâchage et de déchargement
  - Intervalles de vitesse (aller/retour)
  - Distance entre le poste et l'atelier
  - Quantité cible à produire

### 2. Analyse de Performance

- **Métriques de performance** :

  - Utilisation du poste d'enrobé (%)
  - Utilisation de l'atelier (%)
  - Utilisation des camions (%)
  - Total livré vs total posé
  - Durée de la simulation

- **Visualisations interactives** :
  - Graphique du mouvement des camions dans le temps
  - Graphique des taux d'utilisation par composant

### 3. Optimisation de la Flotte

- **Analyse comparative** du nombre optimal de camions
- **Tableaux de résultats** détaillés par configuration
- **Graphiques d'évolution** de l'efficience
- **Détection automatique** du point de stabilisation
- **Recommandations** pour le nombre optimal de camions

## 🚀 Installation

### Prérequis

- Python 3.7+
- pip

### Installation des dépendances

```bash
pip install streamlit simpy numpy pandas plotly
```

## 💻 Utilisation

### Lancement de l'application

```bash
streamlit run app.py
```

### Étapes d'utilisation

1. **Configuration des paramètres de base** :

   - Saisissez la capacité du poste d'enrobé
   - Définissez la capacité et les caractéristiques des camions
   - Ajustez les temps de bâchage et déchargement

2. **Configuration de la production** :

   - Spécifiez le nombre de camions
   - Définissez la distance entre le poste et l'atelier
   - Fixez la quantité cible d'enrobé

3. **Lancement de la simulation** :

   - Cliquez sur "Lancer la simulation"
   - Analysez les résultats et graphiques

4. **Optimisation de la flotte** :
   - Configurez les paramètres d'optimisation
   - Cliquez sur "Afficher les résultats d'optimisation"
   - Consultez le nombre optimal de camions recommandé

## 📊 Interprétation des Résultats

### Métriques Clés

- **Utilisation Poste d'Enrobé** : Pourcentage du temps où le poste produit activement
- **Utilisation Atelier** : Pourcentage du temps où l'atelier pose l'enrobé
- **Utilisation Camions** : Pourcentage moyen des camions en activité

### Optimisation

L'application détermine automatiquement le nombre optimal de camions en analysant le point de stabilisation de l'utilisation de l'atelier, où l'ajout de camions supplémentaires n'améliore plus significativement les performances.

## 🔧 Architecture Technique

### Technologies Utilisées

- **Streamlit** : Interface utilisateur web
- **SimPy** : Moteur de simulation d'événements discrets
- **NumPy/Pandas** : Traitement et analyse des données
- **Plotly** : Visualisations interactives

### Structure du Code

- **AsphaltSimulation** : Classe principale gérant la simulation
- **Processus Camions** : Simulation du cycle de chaque camion
- **Processus Poste** : Simulation de la production d'enrobés
- **Surveillance Système** : Collecte des métriques en temps réel
- **Optimisation Flotte** : Algorithme d'optimisation du nombre de camions

## 📈 Cas d'Usage

### Construction Routière

- Planification de projets de revêtement routier
- Dimensionnement optimal des flottes de camions
- Analyse de la productivité des chantiers

### Optimisation Logistique

- Identification des goulots d'étranglement
- Amélioration de l'efficacité opérationnelle
- Réduction des coûts de transport

### Formation et Analyse

- Simulation de différents scénarios
- Formation des équipes de planification
- Analyse comparative de configurations

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Committez vos changements (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Poussez vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrez une Pull Request

## 📝 License

Ce projet est sous licence [MIT](LICENSE).

## 📞 Support

Pour toute question ou support, veuillez ouvrir une issue sur GitHub.

---

**Développé avec ❤️ pour optimiser la logistique de construction routière**
