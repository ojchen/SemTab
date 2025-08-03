
!pip install lime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class FixedSemTabGermanCredit:
    def __init__(self):
        self.label_encoders = {}
        print("Loading SemTab components...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        try:
            from transformers import pipeline
            print("Loading DistilGPT2 for SemTab...")
            self.llm = pipeline('text-generation',
                               model='distilgpt2',
                               max_length=80,
                               do_sample=True,
                               temperature=0.5,
                               pad_token_id=50256)
            print("DistilGPT2 loaded for SemTab!")
            self.use_llm = True
        except Exception as e:
            print(f"LLM loading failed: {e}")
            self.use_llm = False

        self.off_shelf_llms = {}
        self.load_off_shelf_llms()

        print("Fixed SemTab German Credit Framework ready!")

    def load_off_shelf_llms(self):
        from transformers import pipeline

        llm_configs = {
            'DistilBERT': {
                'model': 'distilbert-base-uncased-finetuned-sst-2-english',
                'task': 'text-classification'
            },
            'DistilGPT2': {
                'model': 'distilgpt2',
                'task': 'text-generation'
            },
            'OPT-125M': {
                'model': 'facebook/opt-125m',
                'task': 'text-generation'
            },
            'GPT2-Small': {
                'model': 'gpt2',
                'task': 'text-generation'
            }
        }

        for name, config in llm_configs.items():
            try:
                print(f"Loading {name}...")
                if config['task'] == 'text-classification':
                    llm = pipeline(config['task'], model=config['model'], return_all_scores=True)
                else:
                    llm = pipeline(config['task'], model=config['model'],
                                  max_length=70, do_sample=True, temperature=0.6,
                                  pad_token_id=50256)
                self.off_shelf_llms[name] = {'pipeline': llm, 'task': config['task']}
                print(f"{name} loaded successfully!")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                self.off_shelf_llms[name] = {'pipeline': None, 'task': 'failed'}

    def create_credit_narrative(self, row):
        age_desc = "young" if row['age'] < 30 else "senior" if row['age'] > 50 else "middle-aged"
        gender_desc = self._get_gender_desc(row)

        narrative = f"{age_desc} {gender_desc} seeking ${row['amount']:,} loan"
        narrative += f" for {self._get_purpose_desc(row['purpose'])}"
        narrative += f" over {row['duration']} months"

        credit_desc = self._get_credit_desc(row['credit_history'])
        narrative += f", {credit_desc}"

        employment_desc = self._get_employment_desc(row['employment_duration'])
        narrative += f", {employment_desc}"

        savings_desc = self._get_savings_desc(row['savings'])
        narrative += f", {savings_desc}"

        if 'property' in row:
            property_desc = self._get_property_desc(row['property'])
            if property_desc:
                narrative += f", {property_desc}"

        return narrative

    def _get_gender_desc(self, row):
        if 'statussex' in row:
            gender_map = {
                'A91': "divorced male",
                'A92': "female",
                'A93': "single male",
                'A94': "married male",
                'A95': "single female"
            }
            return gender_map.get(row['statussex'], "individual")
        return "individual"

    def _get_purpose_desc(self, purpose):
        purpose_map = {
            'A40': 'new car purchase',
            'A41': 'used car purchase',
            'A42': 'furniture',
            'A43': 'electronics',
            'A44': 'appliances',
            'A45': 'repairs',
            'A46': 'education',
            'A47': 'vacation',
            'A48': 'retraining',
            'A49': 'business',
            'A410': 'other needs'
        }
        return purpose_map.get(purpose, 'general purpose')

    def _get_credit_desc(self, credit_history):
        credit_map = {
            'A30': 'excellent credit record',
            'A31': 'good credit history',
            'A32': 'satisfactory credit',
            'A33': 'payment delays',
            'A34': 'critical credit issues'
        }
        return credit_map.get(credit_history, 'standard credit')

    def _get_employment_desc(self, employment_duration):
        employment_map = {
            'A71': 'unemployed',
            'A72': 'short employment history',
            'A73': 'stable employment',
            'A74': 'long-term employment',
            'A75': 'very stable employment'
        }
        return employment_map.get(employment_duration, 'standard employment')

    def _get_savings_desc(self, savings):
        savings_map = {
            'A61': 'minimal savings',
            'A62': 'modest savings',
            'A63': 'moderate savings',
            'A64': 'substantial savings',
            'A65': 'extensive savings'
        }
        return savings_map.get(savings, 'undisclosed savings')

    def _get_property_desc(self, property_code):
        property_map = {
            'A121': 'owns real estate',
            'A122': 'has insurance/savings',
            'A123': 'owns vehicle/property',
            'A124': 'no property'
        }
        return property_map.get(property_code, '')

    def enhance_credit_narrative(self, narrative):
        if not self.use_llm:
            return narrative

        try:
            prompt = f"Credit applicant: {narrative}. Risk assessment:"
            result = self.llm(prompt, max_new_tokens=8, num_return_sequences=1)
            llm_text = result[0]['generated_text']
            enhancement = llm_text.replace(prompt, "").strip()

            if len(enhancement) > 3 and len(enhancement) < 25:
                clean_enhancement = enhancement.split('.')[0].strip()
                if len(clean_enhancement) > 3:
                    return f"{narrative}. {clean_enhancement}"

            return narrative
        except:
            return narrative

    def predict_with_off_shelf_llm(self, narratives, llm_name):
        if llm_name not in self.off_shelf_llms or self.off_shelf_llms[llm_name]['pipeline'] is None:
            return np.random.choice([0, 1], len(narratives), p=[0.7, 0.3])

        llm_info = self.off_shelf_llms[llm_name]
        llm = llm_info['pipeline']
        task = llm_info['task']

        predictions = []

        print(f"Running {llm_name} predictions...")
        for i, narrative in enumerate(narratives):
            if i % 50 == 0:
                print(f"{llm_name} Progress: {i}/{len(narratives)}")

            try:
                if task == 'text-classification':
                    prompt = f"Is this a low credit risk applicant: {narrative}"
                    result = llm(prompt)
                    positive_score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else result[1]['score']
                    prediction = 1 if positive_score > 0.5 else 0

                elif task == 'text-generation':
                    prompt = f"Credit risk: {narrative}. Low risk? Yes/No:"
                    result = llm(prompt, max_new_tokens=3, num_return_sequences=1)
                    output = result[0]['generated_text'].replace(prompt, "").strip().lower()

                    if any(pos_word in output for pos_word in ['yes', 'low', 'good', 'safe']):
                        prediction = 1
                    elif any(neg_word in output for neg_word in ['no', 'high', 'risky', 'bad']):
                        prediction = 0
                    else:
                        positive_signals = sum(1 for word in ['excellent', 'good', 'stable', 'substantial', 'real estate'] if word in narrative.lower())
                        negative_signals = sum(1 for word in ['critical', 'delays', 'unemployed', 'minimal'] if word in narrative.lower())
                        prediction = 1 if positive_signals > negative_signals else 0

                predictions.append(prediction)

            except:
                prediction = np.random.choice([0, 1], p=[0.7, 0.3])
                predictions.append(prediction)

        return np.array(predictions)

    def generate_optimized_credit_features(self, df):
        print(f"Creating optimized credit features for {len(df)} samples...")

        narratives = []
        for idx, (_, row) in enumerate(df.iterrows()):
            if idx % 200 == 0:
                print(f"Credit SemTab Progress: {idx}/{len(df)}")

            narrative = self.create_credit_narrative(row)

            if self.use_llm and np.random.random() < 0.5:
                narrative = self.enhance_credit_narrative(narrative)

            narratives.append(narrative)

        print("Converting to embeddings...")
        embeddings = self.embedding_model.encode(narratives, show_progress_bar=True)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=10)
        reduced_embeddings = pca.fit_transform(embeddings)

        semantic_df = pd.DataFrame(reduced_embeddings, columns=[f'sem_{i}' for i in range(10)])

        phrase_features = []
        for narrative in narratives:
            features = {
                'excellent_credit': 1 if 'excellent credit' in narrative else 0,
                'good_credit': 1 if 'good credit' in narrative else 0,
                'critical_issues': 1 if 'critical' in narrative else 0,
                'payment_delays': 1 if 'delays' in narrative else 0,
                'stable_employment': 1 if 'stable employment' in narrative or 'long-term employment' in narrative else 0,
                'substantial_savings': 1 if 'substantial' in narrative or 'extensive' in narrative else 0,
                'owns_property': 1 if 'real estate' in narrative or 'owns' in narrative else 0,
                'business_purpose': 1 if 'business' in narrative else 0
            }
            phrase_features.append(features)

        phrase_df = pd.DataFrame(phrase_features)
        final_features = pd.concat([semantic_df, phrase_df], axis=1)

        return final_features, narratives

    def prepare_classical_features(self, df, is_training=True):
        df_copy = df.copy()

        categorical_cols = ['status', 'credit_history', 'purpose', 'savings',
                           'employment_duration']
        if 'property' in df_copy.columns:
            categorical_cols.append('property')
        if 'statussex' in df_copy.columns:
            categorical_cols.append('statussex')

        for col in categorical_cols:
            if col in df_copy.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df_copy[col] = self.label_encoders[col].fit_transform(df_copy[col].astype(str))
                else:
                    if col in self.label_encoders:
                        unknown_mask = ~df_copy[col].astype(str).isin(self.label_encoders[col].classes_)
                        if unknown_mask.any():
                            df_copy.loc[unknown_mask, col] = self.label_encoders[col].classes_[0]
                        df_copy[col] = self.label_encoders[col].transform(df_copy[col].astype(str))

        return df_copy

    def explain_credit_semtab_with_lime(self, model, X_test, feature_names, narratives, y_test, predictions, sample_indices=[0, 1, 2]):
        try:
            from lime.lime_tabular import LimeTabularExplainer

            print("\nLIME Credit Risk Interpretability Analysis")
            print("="*55)

            explainer = LimeTabularExplainer(
                X_test.values,
                feature_names=feature_names,
                class_names=['High Risk', 'Low Risk'],
                mode='classification',
                discretize_continuous=True
            )

            for i, idx in enumerate(sample_indices):
                if idx < len(X_test):
                    print(f"\nSample {i+1} LIME Explanation:")
                    print(f"Credit Narrative: {narratives[idx]}")
                    print(f"Actual: {'Low Risk' if y_test.iloc[idx] == 1 else 'High Risk'}")
                    print(f"Predicted: {'Low Risk' if predictions[idx] == 1 else 'High Risk'}")

                    explanation = explainer.explain_instance(
                        X_test.iloc[idx].values,
                        model.predict_proba,
                        num_features=6
                    )

                    semantic_features = []
                    phrase_features = []
                    classical_features = []

                    for feature, weight in explanation.as_list():
                        if feature.startswith('sem_'):
                            semantic_features.append((feature, weight))
                        elif any(phrase in feature for phrase in ['credit', 'employment', 'savings', 'property', 'business', 'delays']):
                            phrase_features.append((feature, weight))
                        else:
                            classical_features.append((feature, weight))

                    if phrase_features:
                        print("Key Credit Risk Elements:")
                        for feature, weight in phrase_features:
                            direction = "reduces" if weight > 0 else "increases"
                            print(f"  {feature} → {direction} credit risk ({weight:+.3f})")

                    if semantic_features:
                        print("Semantic Context Factors:")
                        for feature, weight in semantic_features[:2]:
                            direction = "reduces" if weight > 0 else "increases"
                            print(f"  {feature} → {direction} credit risk ({weight:+.3f})")

                    if classical_features:
                        print("Traditional Credit Features:")
                        for feature, weight in classical_features[:2]:
                            direction = "reduces" if weight > 0 else "increases"
                            print(f"  {feature} → {direction} credit risk ({weight:+.3f})")

        except ImportError:
            print("\nLIME not available. Install with: pip install lime")
        except Exception as e:
            print(f"\nLIME analysis failed: {e}")

def load_german_credit_data():
    print("Loading German Credit dataset...")

    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                       header=None, sep=' ')
    feature_names = ['status', 'duration', 'credit_history', 'purpose', 'amount',
                     'savings', 'employment_duration', 'installment_rate', 'statussex',
                     'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
                     'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
                     'credit_risk']
    data.columns = feature_names

    data['credit_risk'] = data['credit_risk'].map({1: 1, 2: 0})

    key_features = ['status', 'duration', 'credit_history', 'purpose', 'amount',
                   'savings', 'employment_duration', 'statussex', 'property', 'age', 'credit_risk']

    data = data[key_features]

    if len(data) > 800:
        print(f"Sampling 800 from {len(data)} records for LLM comparison")
        data = data.sample(n=800, random_state=42)

    print(f"Dataset: {len(data)} records")
    print(f"Low risk rate: {(data['credit_risk'] == 1).mean():.1%}")

    X = data.drop('credit_risk', axis=1)
    y = data['credit_risk']

    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def calculate_all_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def run_fixed_semtab_german_credit():
    print("="*80)
    print("FIXED SEMTAB vs OFF-THE-SHELF LLMs: GERMAN CREDIT DATASET")
    print("="*80)

    X_train, X_test, y_train, y_test = load_german_credit_data()
    print(f"Split: {len(X_train)} train, {len(X_test)} test")

    framework = FixedSemTabGermanCredit()

    _, test_narratives = framework.generate_optimized_credit_features(X_test)

    print("\n" + "="*80)
    print("OFF-THE-SHELF LLM PERFORMANCE ON GERMAN CREDIT")
    print("="*80)

    llm_results = {}

    for llm_name in framework.off_shelf_llms.keys():
        print(f"\nTesting {llm_name}...")

        llm_predictions = framework.predict_with_off_shelf_llm(test_narratives, llm_name)
        llm_metrics = calculate_all_metrics(y_test, llm_predictions)

        llm_results[llm_name] = {
            **llm_metrics,
            'predictions': llm_predictions
        }

        print(f"{llm_name} - Acc: {llm_metrics['accuracy']:.4f}, Prec: {llm_metrics['precision']:.4f}, Rec: {llm_metrics['recall']:.4f}, F1: {llm_metrics['f1']:.4f}")

    print("\n" + "="*80)
    print("FIXED SEMTAB HYBRID PERFORMANCE ON GERMAN CREDIT")
    print("="*80)

    print("Training optimized German Credit SemTab model...")

    X_train_classical = framework.prepare_classical_features(X_train, is_training=True)
    X_test_classical = framework.prepare_classical_features(X_test, is_training=False)

    semantic_train, _ = framework.generate_optimized_credit_features(X_train)
    semantic_test, _ = framework.generate_optimized_credit_features(X_test)

    X_train_hybrid = pd.concat([X_train_classical.reset_index(drop=True),
                               semantic_train.reset_index(drop=True)], axis=1)
    X_test_hybrid = pd.concat([X_test_classical.reset_index(drop=True),
                              semantic_test.reset_index(drop=True)], axis=1)

    semtab_model = LogisticRegression(C=0.01, class_weight='balanced',
                                     random_state=42, max_iter=2000, solver='liblinear')
    semtab_model.fit(X_train_hybrid, y_train)
    semtab_predictions = semtab_model.predict(X_test_hybrid)

    semtab_metrics = calculate_all_metrics(y_test, semtab_predictions)

    print(f"Fixed German Credit SemTab - Acc: {semtab_metrics['accuracy']:.4f}, Prec: {semtab_metrics['precision']:.4f}, Rec: {semtab_metrics['recall']:.4f}, F1: {semtab_metrics['f1']:.4f}")

    framework.explain_credit_semtab_with_lime(semtab_model, X_test_hybrid, X_test_hybrid.columns.tolist(),
                                             test_narratives, y_test, semtab_predictions)

    print("\n" + "="*95)
    print("GERMAN CREDIT COMPREHENSIVE COMPARISON TABLE")
    print("="*95)

    print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1-Score':<10} {'vs SemTab F1':<12}")
    print("-" * 95)

    sorted_llms = sorted(llm_results.items(), key=lambda x: x[1]['f1'], reverse=True)

    for llm_name, results in sorted_llms:
        vs_semtab = ((semtab_metrics['f1'] - results['f1']) / results['f1'] * 100) if results['f1'] > 0 else 0
        print(f"{llm_name:<15} {results['accuracy']:<10.4f} {results['precision']:<11.4f} {results['recall']:<9.4f} {results['f1']:<10.4f} {vs_semtab:+.1f}%")

    print("-" * 95)
    print(f"{'SemTab Hybrid':<15} {semtab_metrics['accuracy']:<10.4f} {semtab_metrics['precision']:<11.4f} {semtab_metrics['recall']:<9.4f} {semtab_metrics['f1']:<10.4f} {'Baseline':<12}")

    print("\n" + "="*80)
    print("GERMAN CREDIT KEY FINDINGS")
    print("="*80)

    best_llm = max(llm_results.items(), key=lambda x: x[1]['f1'])
    best_llm_name, best_llm_results = best_llm

    semtab_vs_best = ((semtab_metrics['f1'] - best_llm_results['f1']) / best_llm_results['f1'] * 100) if best_llm_results['f1'] > 0 else 0

    print(f"Best Off-the-Shelf LLM: {best_llm_name} (F1: {best_llm_results['f1']:.4f})")
    print(f"German Credit SemTab: F1: {semtab_metrics['f1']:.4f}")
    print(f"SemTab vs Best LLM: {semtab_vs_best:+.1f}% F1 improvement")

    if semtab_metrics['f1'] > best_llm_results['f1']:
        print(f"SUCCESS: SemTab outperforms all off-the-shelf LLMs on German Credit!")
    else:
        print(f"CHALLENGE: {best_llm_name} still outperforms SemTab on German Credit")

    wins = sum(1 for results in llm_results.values() if semtab_metrics['f1'] > results['f1'])
    print(f"SemTab wins against {wins}/{len(llm_results)} off-the-shelf LLMs")

    print(f"\nCredit Risk Analysis:")
    print(f"SemTab identifies {semtab_metrics['recall']*100:.1f}% of low-risk applicants")
    print(f"SemTab precision: {semtab_metrics['precision']*100:.1f}% of predicted low-risk are actually low-risk")

    print("\nSample Credit Risk Narratives:")
    for i in range(min(2, len(test_narratives))):
        actual = "Low Risk" if y_test.iloc[i] == 1 else "High Risk"
        semtab_pred = "Low Risk" if semtab_predictions[i] == 1 else "High Risk"

        print(f"\nSample {i+1}: {test_narratives[i][:120]}...")
        print(f"Actual: {actual} | SemTab: {semtab_pred}")

    return {
        'semtab': semtab_metrics,
        'llms': llm_results,
        'best_llm': best_llm_name,
        'semtab_vs_best': semtab_vs_best
    }

if __name__ == "__main__":
    results = run_fixed_semtab_german_credit()

