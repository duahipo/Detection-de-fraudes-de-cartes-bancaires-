# Detection-de-fraudes-de-cartes-bancaires-
Détection de fraudes de cartes bancaires   (DATA MINING)      
<img width="597" height="455" alt="image" src="https://github.com/user-attachments/assets/c60d0c7a-cb5e-4a12-b6a1-e0bea7c581a1" />



<img width="965" height="876" alt="image" src="https://github.com/user-attachments/assets/d5bf12a2-eb71-4a70-bd92-ddbef7803c0e" />



<img width="520" height="455" alt="image" src="https://github.com/user-attachments/assets/e467c918-d274-46c0-b369-a360422a6936" />




<img width="554" height="435" alt="image" src="https://github.com/user-attachments/assets/dea815e2-e13c-4f38-b55a-0587ec3999cb" />




<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/fb9f85cc-f71e-4d5f-b58d-bb301a9f9047" />





<img width="698" height="547" alt="image" src="https://github.com/user-attachments/assets/61ce2208-f04a-44a6-96f6-9f26a0a4a646" />




#   Projet Data Mining — Détection de Fraude Bancaire

##   Interprétation Métier & Rapport Final

---

## 1.  Pourquoi la Recall est plus importante que l’Accuracy ?

Dans les systèmes financiers, le taux de fraude est extrêmement faible (<0,2 % des transactions), ce qui crée un **déséquilibre des classes**.

###  Limite de l’Accuracy
Un modèle qui prédit toujours *"non frauduleuse"* peut atteindre **99,8 % d’accuracy**, mais :
- Il ne détecte **aucune fraude**
- Il est **inutile en pratique**

### Importance de la Recall (Sensibilité)
La Recall mesure la proportion de fraudes détectées :

\[
Recall = \frac{\text{Fraudes détectées}}{\text{Fraudes totales}}
\]

### Raison métier
- Chaque fraude non détectée = **perte financière directe**
- Les **faux négatifs** coûtent beaucoup plus cher que les faux positifs

###  Conclusion métier
- Une **Recall élevée** protège la banque et ses clients, même avec quelques fausses alertes.

---

## 2. Meilleur modèle : Performance vs Métier

| Modèle              | Recall        | Precision     | ROC-AUC     | Commentaire |
|--------------------|--------------|--------------|------------|------------|
| Logistic Regression | Moyenne      | Moyenne      | Bonne      | Simple mais limité |
| Decision Tree       | Élevée mais instable | Moyenne | Variable   | Risque d’overfitting |
| Random Forest       | Très élevée  | Bonne        | Excellente | ⭐ Meilleur compromis |
| Isolation Forest    | Mauvais      | n/a          | Correct    | Non supervisé |
| K-Means             | Non supervisé| n/a          | Faible     | Pas adapté seul |

###  Recommandation métier
- **Random Forest** → Production  
- **XGBoost** → Optimisation (après tuning)

### ✔️ Avantages
- Haute Recall  
- Bonne Precision  
- Robustesse aux données déséquilibrées  
- Interprétabilité (importance des variables)

---

## 3.  Valeur métier du Clustering (K-Means)

Le clustering ne détecte pas directement la fraude, mais apporte :

###  Segmentation des comportements
- Paiements fréquents / faibles montants  
- Achats irréguliers / montants élevés  
- Usage intensif en ligne  
- Transactions nocturnes  

###  Détection d’anomalies
- Identification d’outliers  
- Pré-alertes  
- Surveillance renforcée  

###  Amélioration du ML
- Feature engineering (cluster = variable)  
- Amélioration des performances des modèles  

###  Conclusion
- Comprendre le comportement normal = mieux détecter les anomalies

---

## 4.  Règles d’association (Apriori)

###  Exemples utiles
- Transaction nocturne + montant élevé → risque élevé  
- Transactions rapides (<10 sec) → suspicion  
- Pays inhabituel → alerte  
- Plusieurs échecs → comportement anormal  

###  Valeur métier
- Création de règles antifraude  
- Déclenchement de vérifications (OTP, 3D Secure)  
- Formation des équipes sécurité  

### Conclusion
- Complément du ML avec une logique explicable et traçable

---

## 5. Pipeline ML (Système opérationnel)

### Architecture

#### 1. Collecte des données
- API interne : montant, lieu, heure, client ID, device ID, IP, historique

#### 2. Prétraitement
- Normalisation / scaling  
- Feature engineering  
- Enrichissement (historique, clusters)

#### 3. Modèle ML
- Random Forest / XGBoost  
- Détection < 100 ms  
- Sortie : probabilité de fraude  

#### 4. Prise de décision

| Score        | Action |
|-------------|-------|
| < 0.2       | Acceptée |
| 0.2 – 0.5   | Vérification (OTP) |
| > 0.5       | Bloquée + alerte |

#### 5. Monitoring
- Tests A/B  
- Détection de drift  
- Ré-entraînement (hebdo/mensuel)  
- Dashboard (Recall, pertes évitées)

#### 6. Audit & traçabilité
- Journalisation des décisions  
- Explication des prédictions  

###  Conclusion
- Pipeline fiable, rapide et conforme (PSD2, KYC, AML)

---

##  Conclusion finale

- ✔️ La **Recall est prioritaire** pour limiter les pertes  
- ✔️ Meilleurs modèles : **Random Forest / XGBoost**  
- ✔️ Le clustering aide à comprendre les comportements  
- ✔️ Les règles d’association apportent une logique métier  
- ✔️ Pipeline temps réel indispensable  

---

##  Validation du projet Data Mining

| Étape                         | Statut |
|------------------------------|--------|
| Définition des objectifs     | ✔️ |
| Préparation des données (KDD)| ✔️ |
| Fouille de données           | ✔️ |
| Évaluation des modèles       | ✔️ |

###   Conclusion générale
=> Toutes les étapes sont validées.  
=> Le notebook final peut être généré selon cette structure.

---
