import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sympy import primerange

# Clean memory
gc.collect()

# Set working directory
os.chdir("C:/Users/Sebastian/OneDrive/Escritorio/DataMining/DMEco")

# Create an artificial dataset
n_samples = 1000
n_features = 20
np.random.seed(42)

data = np.random.rand(n_samples, n_features)
target = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])

columns = [f'feature_{i}' for i in range(n_features)]
df_train = pd.DataFrame(data, columns=columns)
df_train['clase_ternaria'] = target

data_apply = np.random.rand(n_samples, n_features)
df_apply = pd.DataFrame(data_apply, columns=columns)
df_apply['numero_de_cliente'] = np.arange(n_samples)

df_train.to_csv('artificial_train.csv', index=False)
df_apply.to_csv('artificial_apply.csv', index=False)

# Load the dataset
dtrain = pd.read_csv("artificial_train.csv")
dapply = pd.read_csv("artificial_apply.csv")

# Add 30 random canaritos
for i in range(1, 31):
    dtrain[f'canarito{i}'] = np.random.rand(dtrain.shape[0])
    dapply[f'canarito{i}'] = np.random.rand(dapply.shape[0])

# Reorder columns so canaritos come first
original_columns = df_train.columns.tolist()
new_columns = [col for col in dtrain.columns if col not in original_columns] + original_columns
dtrain = dtrain[new_columns]

# Train a Decision Tree model without limits (using scikit-learn)
X = dtrain.drop(columns=['clase_ternaria'])
y = dtrain['clase_ternaria']
model = DecisionTreeClassifier(random_state=10219, max_depth=30, min_samples_split=2, min_samples_leaf=1)
model.fit(X, y)

# Pruning canaritos: replace impurity reduction of canaritos with a large negative value
prune_threshold = -666
for i in range(1, 31):
    model.tree_.impurity[X.columns.get_loc(f'canarito{i}')] = prune_threshold

# Predict on dapply dataset
predictions = model.predict_proba(dapply.drop(columns=['numero_de_cliente']))[:, 2]

# Create output DataFrame
entrega = pd.DataFrame({
    'numero_de_cliente': dapply['numero_de_cliente'],
    'Predicted': (predictions > 1/60).astype(int)
})

# Create directories to save results
os.makedirs("./labo/exp/KA5230/", exist_ok=True)
os.chdir("C:/Users/Sebastian/OneDrive/Escritorio/DataMining/DMEco/labo/exp/KA5230/")

# Save the results for Kaggle
entrega.to_csv("stopping_at_canaritos.csv", index=False, sep=",")

# Plot and save the original tree
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=model.classes_)
plt.savefig("arbol_libre.pdf")

# Plot and save the pruned tree
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=model.classes_)
plt.savefig("stopping_at_canaritos.pdf")

# Close all plots
plt.close('all')

# Load final dataset
dataset = dtrain.copy()
dataset['clase01'] = dataset['clase_ternaria'].apply(lambda x: 1 if x in [1, 2] else 0)

# Generate primes for seeds
primos = list(primerange(100000, 1000000))
np.random.seed(10219)
ksemillas = np.random.choice(primos, 20, replace=False)

# Model parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train models with different seeds and save them
for seed in ksemillas:
    print(f"Training with seed {seed}")
    params['seed'] = seed
    dtrain_lgb = lgb.Dataset(data=dataset.drop(columns=['clase01', 'clase_ternaria']),
                             label=dataset['clase01'])
    model = lgb.train(params, dtrain_lgb, num_boost_round=100)
    model.save_model(f"lightgbm_model_seed_{seed}.txt")

    # Create feature importance
    importance = model.feature_importance(importance_type='split')
    feature_names = model.feature_name()
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=False)
    feature_importance_df.to_csv(f"feature_importance_seed_{seed}.csv", index=False)

    # Prediction on future data
    future_data = dapply.copy()
    future_data['prediction'] = model.predict(future_data.drop(columns=['numero_de_cliente']))
    future_data[['numero_de_cliente', 'prediction']].to_csv(f"predictions_seed_{seed}.csv", index=False)

# Hybridize Predictions
def hybridize_predictions(exp_dir, tb_catalogo, PARAM, ksemillas):
    tb_final = None

    for i in range(len(tb_catalogo)):
        exp_carpeta = os.path.join(exp_dir, tb_catalogo['experiment'][i])
        semillerios = [f for f in os.listdir(exp_carpeta) if PARAM['files']['output']['prefijo_pred_semillerio'] in f]

        tb_predicciones = pd.read_csv(tb_catalogo['value'][i], sep="\t")

        for archivo in tb_predicciones[tb_predicciones['archivo'].str.contains('semillerio')]['archivo']:
            tb_prediccion = pd.read_csv(os.path.join(exp_carpeta, archivo), sep="\t")

            if tb_final is None:
                tb_final = tb_prediccion.copy()
            else:
                tb_final['pred_acumulada'] += tb_prediccion['pred_acumulada']

    tb_final.to_csv("futuro_prediccion_hibridacion.csv", index=False, sep=",")

    tb_predicciones = pd.DataFrame({
        'archivo': ["futuro_prediccion_hibridacion.csv"],
        'iteracion_bayesiana': [-1],
        'ganancia': [-1]
    })
    tb_predicciones.to_csv(PARAM['files']['output']['tb_predicciones'], index=False, sep="\t")

    # Generate submission files based on thresholds
    cortes = range(PARAM['KA_start'], PARAM['KA_end'] + 1, PARAM['KA_step'])
    tb_final.sort_values(by='pred_acumulada', ascending=False, inplace=True)

    for corte in cortes:
        tb_final['Predicted'] = 0
        tb_final.iloc[:corte, tb_final.columns.get_loc('Predicted')] = 1
        nom_submit = f"hibridacion_{corte:05d}.csv"
        tb_final[['numero_de_cliente', 'Predicted']].to_csv(nom_submit, index=False, sep=",")

# Example usage
exp_dir = "./labo/exp/"
tb_catalogo = pd.DataFrame({
    'key': ['predicciones', 'predicciones'],
    'value': ['predictions_seed_1.csv', 'predictions_seed_2.csv'],
    'experiment': ['KA5230', 'KA5231']
})
PARAM = {
    'files': {
        'output': {
            'prefijo_pred_semillerio': 'predictions_seed_',
            'tb_predicciones': 'tb_predicciones.csv',
            'tb_submits': 'tb_submits.csv'
        }
    },
    'KA_start': 100,
    'KA_end': 900,
    'KA_step': 100
}

hybridize_predictions(exp_dir, tb_catalogo, PARAM, ksemillas)
