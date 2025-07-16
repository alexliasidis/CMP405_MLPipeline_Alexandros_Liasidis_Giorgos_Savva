# CMP405_MLPipeline_Alexandros_Liasidis_Giorgos_Savva
# Student Performance Prediction with AI Agent

Αυτό το έργο υλοποιεί ένα πλήρες Machine Learning pipeline για την πρόβλεψη της επιτυχίας μαθητών, με χρήση πραγματικών δεδομένων εξετάσεων. Περιλαμβάνει την εκπαίδευση ενός μοντέλου ταξινόμησης και τη δημιουργία ενός διαδραστικού AI Agent που μπορεί να απαντά σε ερωτήσεις σχετικά με νέους μαθητές.

---

## Dataset

Το dataset προέρχεται από το Kaggle:  
[Students Performance in Exams] (https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

Περιλαμβάνει:
- Δημογραφικά στοιχεία: φύλο, εθνικότητα
- Εκπαιδευτικό υπόβαθρο γονέων
- Είδος lunch & test preparation course
- Βαθμολογίες σε Math, Reading, Writing (0–100)

---

## Τι Μοντέλο Φτιάξαμε

- Χρησιμοποιήσαμε Random Forest Classifier
- Ορίσαμε ως στόχο (`pass = 1`) τους μαθητές με μέσο όρο ≥ 60
- Έγινε προεπεξεργασία, κωδικοποίηση κατηγορικών χαρακτηριστικών και αξιολόγηση του μοντέλου με:
  - Accuracy
  - Precision / Recall / F1
  - Confusion Matrix

---

## Τι Κάνει ο AI Agent

Το αρχείο `student_agent.py` είναι ένας διαδραστικός agent που:

1. Φορτώνει το αποθηκευμένο μοντέλο:
	model = joblib.load('student_performance_model.pkl')

2. Περιμένει input από τον χρήστη:
	- gender, ethnicity, parental education, lunch, test prep
	- math, reading, writing score

3. Κάνει encoding (με mapping ίδιο με το training):
	label_maps = { "gender": {"female": 0, "male": 1}, ... }

4. Δημιουργεί DataFrame με τα χαρακτηριστικά:
	input_df = pd.DataFrame([encoded_values], columns=feature_names)

5. Εκτελεί πρόβλεψη:
	prediction = model.predict(input_df)[0]

6. Εμφανίζει αποτέλεσμα:
	✅ Pass ή ❌ Fail

---

## 🖼️ Screenshots

### Εκπαίδευση Μοντέλου (Confusion Matrix)
![confusion-matrix](images/confusion_matrix.png)

### Agent Πρόβλεψη
![agent-predict](images/agent_predict.png)

---

## Τεχνολογίες
- Python 3.11
- Pandas, scikit-learn, joblib
- Matplotlib / Seaborn (για EDA)
- CLI interface (agent)
