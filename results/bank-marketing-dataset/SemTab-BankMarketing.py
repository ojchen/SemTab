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

class FixedSemTabVsOffShelfLLMs:
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
        
        print("Fixed SemTab vs Off-the-Shelf LLMs Framework ready!")
    
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
    
    def create_focused_narrative(self, row):
        age_desc = "young" if row['age'] < 30 else "senior" if row['age'] > 60 else "middle-aged"
        
        narrative = f"{age_desc} {row['marital']} {row['job']} with {row['education']}"
        
        if row['housing'] == 'yes' and row['loan'] == 'yes':
            narrative += ", multiple loans"
        elif row['housing'] == 'yes':
            narrative += ", homeowner with mortgage"
        elif row['loan'] == 'yes':
            narrative += ", has personal loan"
        else:
            narrative += ", debt-free"
        
        if row['duration'] > 400:
            narrative += f", very engaged {row['duration']}s call"
        elif row['duration'] > 200:
            narrative += f", engaged {row['duration']}s call"
        else:
            narrative += f", brief {row['duration']}s call"
        
        narrative += f" via {row['contact']}"
        
        if row['previous'] > 0:
            if row['poutcome'] == 'success':
                narrative += f", {row['previous']} prior wins"
            elif row['poutcome'] == 'failure':
                narrative += f", {row['previous']} prior losses"
        
        return narrative
    
    def enhance_narrative_aggressively(self, narrative):
        if not self.use_llm:
            return narrative
        
        try:
            prompt = f"Customer: {narrative}. Likely to subscribe:"
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
            return np.random.choice([0, 1], len(narratives), p=[0.885, 0.115])
        
        llm_info = self.off_shelf_llms[llm_name]
        llm = llm_info['pipeline']
        task = llm_info['task']
        
        predictions = []
        
        print(f"Running {llm_name} predictions...")
        for i, narrative in enumerate(narratives):
            if i % 100 == 0:
                print(f"{llm_name} Progress: {i}/{len(narratives)}")
            
            try:
                if task == 'text-classification':
                    prompt = f"Will subscribe to term deposit: {narrative}"
                    result = llm(prompt)
                    positive_score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else result[1]['score']
                    prediction = 1 if positive_score > 0.5 else 0
                    
                elif task == 'text-generation':
                    prompt = f"Customer: {narrative}. Subscribe? Yes/No:"
                    result = llm(prompt, max_new_tokens=3, num_return_sequences=1)
                    output = result[0]['generated_text'].replace(prompt, "").strip().lower()
                    
                    if any(pos_word in output for pos_word in ['yes', 'likely', 'subscribe', 'will']):
                        prediction = 1
                    elif any(neg_word in output for neg_word in ['no', 'unlikely', 'won\'t']):
                        prediction = 0
                    else:
                        positive_signals = sum(1 for word in ['engaged', 'university', 'management', 'debt-free', 'wins'] if word in narrative.lower())
                        negative_signals = sum(1 for word in ['brief', 'loans', 'losses', 'unemployed'] if word in narrative.lower())
                        prediction = 1 if positive_signals > negative_signals else 0
                
                predictions.append(prediction)
                
            except:
                prediction = np.random.choice([0, 1], p=[0.885, 0.115])
                predictions.append(prediction)
        
        return np.array(predictions)
    
    def generate_optimized_semtab_features(self, df):
        print(f"Creating optimized SemTab features for {len(df)} samples...")
        
        narratives = []
        for idx, (_, row) in enumerate(df.iterrows()):
            if idx % 300 == 0:
                print(f"SemTab Progress: {idx}/{len(df)}")
            
            narrative = self.create_focused_narrative(row)
            
            if self.use_llm and np.random.random() < 0.5:
                narrative = self.enhance_narrative_aggressively(narrative)
            
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
                'very_engaged': 1 if 'very engaged' in narrative else 0,
                'debt_free': 1 if 'debt-free' in narrative else 0,
                'multiple_loans': 1 if 'multiple loans' in narrative else 0,
                'prior_wins': 1 if 'prior wins' in narrative else 0,
                'prior_losses': 1 if 'prior losses' in narrative else 0,
                'university_educated': 1 if 'university' in narrative else 0,
                'professional_job': 1 if any(job in narrative for job in ['management', 'admin', 'technician']) else 0,
                'homeowner': 1 if 'homeowner' in narrative else 0
            }
            phrase_features.append(features)
        
        phrase_df = pd.DataFrame(phrase_features)
        final_features = pd.concat([semantic_df, phrase_df], axis=1)
        
        return final_features, narratives
    
    def prepare_classical_features(self, df, is_training=True):
        df_copy = df.copy()
        
        categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                           'contact', 'month', 'poutcome']
        
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
    
    def explain_semtab_with_lime(self, model, X_test, feature_names, narratives, y_test, predictions, sample_indices=[0, 1, 2]):
        try:
            from lime.lime_tabular import LimeTabularExplainer
            
            print("\nLIME Interpretability Analysis")
            print("="*50)
            
            explainer = LimeTabularExplainer(
                X_test.values,
                feature_names=feature_names,
                class_names=['Not Subscribe', 'Subscribe'],
                mode='classification',
                discretize_continuous=True
            )
            
            for i, idx in enumerate(sample_indices):
                if idx < len(X_test):
                    print(f"\nSample {i+1} LIME Explanation:")
                    print(f"Narrative: {narratives[idx]}")
                    print(f"Actual: {'Subscribe' if y_test.iloc[idx] == 1 else 'Not Subscribe'}")
                    print(f"Predicted: {'Subscribe' if predictions[idx] == 1 else 'Not Subscribe'}")
                    
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
                        elif any(phrase in feature for phrase in ['engaged', 'debt', 'loans', 'wins', 'losses', 'university', 'professional', 'homeowner']):
                            phrase_features.append((feature, weight))
                        else:
                            classical_features.append((feature, weight))
                    
                    if phrase_features:
                        print("Key Narrative Elements:")
                        for feature, weight in phrase_features:
                            direction = "promotes" if weight > 0 else "reduces"
                            print(f"  {feature} → {direction} subscription ({weight:+.3f})")
                    
                    if semantic_features:
                        print("Semantic Factors:")
                        for feature, weight in semantic_features[:2]:
                            direction = "promotes" if weight > 0 else "reduces"
                            print(f"  {feature} → {direction} subscription ({weight:+.3f})")
                    
                    if classical_features:
                        print("Traditional Features:")
                        for feature, weight in classical_features[:2]:
                            direction = "promotes" if weight > 0 else "reduces"
                            print(f"  {feature} → {direction} subscription ({weight:+.3f})")
            
        except ImportError:
            print("\nLIME not available. Install with: pip install lime")
        except Exception as e:
            print(f"\nLIME analysis failed: {e}")

def load_bank_data():
    print("Loading Bank Marketing dataset...")
    
    data = pd.read_csv('/content/bank-marketing-dataset.csv', sep=';')
    print("Loaded from uploaded file")
    
    key_features = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                   'contact', 'month', 'duration', 'campaign', 'previous', 'poutcome', 'y']
    
    data = data[key_features]
    
    if len(data) > 1500:
        print(f"Sampling 1500 from {len(data)} records for comprehensive LLM comparison")
        data = data.sample(n=1500, random_state=42)
    
    print(f"Dataset: {len(data)} records")
    print(f"Subscription rate: {(data['y'] == 'yes').mean():.1%}")
    
    X = data.drop('y', axis=1)
    y = (data['y'] == 'yes').astype(int)
    
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def calculate_all_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def run_fixed_semtab_vs_off_shelf():
    print("="*80)
    print("FIXED SEMTAB vs OFF-THE-SHELF LLMs COMPREHENSIVE COMPARISON")
    print("="*80)
    
    X_train, X_test, y_train, y_test = load_bank_data()
    print(f"Split: {len(X_train)} train, {len(X_test)} test")
    
    framework = FixedSemTabVsOffShelfLLMs()
    
    _, test_narratives = framework.generate_optimized_semtab_features(X_test)
    
    print("\n" + "="*80)
    print("OFF-THE-SHELF LLM PERFORMANCE")
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
    print("FIXED SEMTAB HYBRID PERFORMANCE")
    print("="*80)
    
    print("Training optimized SemTab hybrid model...")
    
    X_train_classical = framework.prepare_classical_features(X_train, is_training=True)
    X_test_classical = framework.prepare_classical_features(X_test, is_training=False)
    
    semantic_train, _ = framework.generate_optimized_semtab_features(X_train)
    semantic_test, _ = framework.generate_optimized_semtab_features(X_test)
    
    X_train_hybrid = pd.concat([X_train_classical.reset_index(drop=True), 
                               semantic_train.reset_index(drop=True)], axis=1)
    X_test_hybrid = pd.concat([X_test_classical.reset_index(drop=True), 
                              semantic_test.reset_index(drop=True)], axis=1)
    
    semtab_model = LogisticRegression(C=0.01, class_weight='balanced', 
                                     random_state=42, max_iter=2000, solver='liblinear')
    semtab_model.fit(X_train_hybrid, y_train)
    semtab_predictions = semtab_model.predict(X_test_hybrid)
    
    semtab_metrics = calculate_all_metrics(y_test, semtab_predictions)
    
    print(f"Fixed SemTab - Acc: {semtab_metrics['accuracy']:.4f}, Prec: {semtab_metrics['precision']:.4f}, Rec: {semtab_metrics['recall']:.4f}, F1: {semtab_metrics['f1']:.4f}")
    
    framework.explain_semtab_with_lime(semtab_model, X_test_hybrid, X_test_hybrid.columns.tolist(), 
                                      test_narratives, y_test, semtab_predictions)
    
    print("\n" + "="*90)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*90)
    
    print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1-Score':<10} {'vs SemTab F1':<12}")
    print("-" * 90)
    
    sorted_llms = sorted(llm_results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for llm_name, results in sorted_llms:
        vs_semtab = ((semtab_metrics['f1'] - results['f1']) / results['f1'] * 100) if results['f1'] > 0 else 0
        print(f"{llm_name:<15} {results['accuracy']:<10.4f} {results['precision']:<11.4f} {results['recall']:<9.4f} {results['f1']:<10.4f} {vs_semtab:+.1f}%")
    
    print("-" * 90)
    print(f"{'SemTab Hybrid':<15} {semtab_metrics['accuracy']:<10.4f} {semtab_metrics['precision']:<11.4f} {semtab_metrics['recall']:<9.4f} {semtab_metrics['f1']:<10.4f} {'Baseline':<12}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    best_llm = max(llm_results.items(), key=lambda x: x[1]['f1'])
    best_llm_name, best_llm_results = best_llm
    
    semtab_vs_best = ((semtab_metrics['f1'] - best_llm_results['f1']) / best_llm_results['f1'] * 100) if best_llm_results['f1'] > 0 else 0
    
    print(f"Best Off-the-Shelf LLM: {best_llm_name} (F1: {best_llm_results['f1']:.4f})")
    print(f"Fixed SemTab Hybrid: F1: {semtab_metrics['f1']:.4f}")
    print(f"SemTab vs Best LLM: {semtab_vs_best:+.1f}% F1 improvement")
    
    if semtab_metrics['f1'] > best_llm_results['f1']:
        print(f"SUCCESS: Fixed SemTab outperforms all off-the-shelf LLMs!")
    else:
        print(f"CHALLENGE: {best_llm_name} still outperforms SemTab")
    
    wins = sum(1 for results in llm_results.values() if semtab_metrics['f1'] > results['f1'])
    print(f"SemTab wins against {wins}/{len(llm_results)} off-the-shelf LLMs")
    
    print(f"\nRecall Analysis:")
    print(f"SemTab Recall: {semtab_metrics['recall']:.4f} (catching {semtab_metrics['recall']*100:.1f}% of subscribers)")
    print(f"Best LLM Recall: {best_llm_results['recall']:.4f} (catching {best_llm_results['recall']*100:.1f}% of subscribers)")
    
    return {
        'semtab': semtab_metrics,
        'llms': llm_results,
        'best_llm': best_llm_name,
        'semtab_vs_best': semtab_vs_best
    }

if __name__ == "__main__":
    results = run_fixed_semtab_vs_off_shelf()