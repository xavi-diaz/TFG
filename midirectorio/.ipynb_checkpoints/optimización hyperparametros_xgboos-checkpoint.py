import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# 1. Cargar un subconjunto de la base de datos
df = pd.read_csv('scaled_df.csv')
df = df.drop(columns=['source', 'smiles', 'solvent'], errors='ignore')
df = df.dropna()

#identificar las variables 
target = 'peakwavs_max'
features = [col for col in df.columns if col != target]

#División de datos en train y test 
x = df[features]
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Seleccionar un subconjunto de los datos (por ejemplo, el 50% de los datos)
X_sub, _, y_sub, _ = train_test_split(X, y, train_size=0.5, random_state=42)

# 2. Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    'max_depth': [3, 5, 7, 9],  # Profundidad máxima del árbol
    'learning_rate': [0.01, 0.1, 0.2],  # Tasa de aprendizaje
    'n_estimators': [100, 200, 300, 350],  # Número de árboles
    'subsample': [0.8, 1.0],  # Submuestra de datos para cada árbol
    'colsample_bytree': [0.8, 1.0],  # Submuestra de características por árbol
    'gamma': [0, 0.1, 0.2],  # Reducción mínima de pérdida para hacer una partición
}

# 3. Configurar Grid Search
model = xgb.XGBClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

# 4. Entrenar el modelo con Grid Search
grid_search.fit(X_sub, y_sub)

# 5. Mostrar los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

# 6. Evaluar el modelo con los mejores hiperparámetros
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_sub)
accuracy = accuracy_score(y_sub, y_pred)
print(f"Precisión del modelo con los mejores hiperparámetros: {accuracy:.4f}")