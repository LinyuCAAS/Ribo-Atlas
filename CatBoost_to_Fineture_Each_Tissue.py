import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import os

DATA_PATH = "../1.CatBoost/Features.csv"
GLOBAL_MODEL_PATH = "../1.CatBoost/Multi_Tissues_CatBoost_uORF_model.cbm"
FINE_TUNE_MODEL_PATH = "CatBoost_uORF_model_Liver.cbm"
TISSUE_NAME = "Liver"


global_model = CatBoostRegressor()
global_model.load_model(GLOBAL_MODEL_PATH)


df = pd.read_csv(DATA_PATH)

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"Samples: {len(df)}")

df["tissue"] = df["tissue"].astype("category")

X = df.drop(["periodic"], axis=1)
y = df["periodic"]
cat_features = ["tissue"]

single_tissue_df = df[df["tissue"] == TISSUE_NAME]
X_single = single_tissue_df.drop(["periodic"], axis=1)
y_single = single_tissue_df["periodic"]

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_single, y_single, test_size=0.2, random_state=42
)


fine_tune_model = CatBoostRegressor(
    iterations=5000,
    depth=6,
    learning_rate=0.01,
    loss_function='RMSE',
    eval_metric='R2',
    random_seed=42,
    od_type='Iter',
    od_wait=20,
    thread_count=8,
    verbose=100
)


fine_tune_model.fit(
    X_train_t, y_train_t,
    eval_set=(X_test_t, y_test_t),
    cat_features=cat_features,
    init_model=global_model
)

fine_tune_model.save_model(FINE_TUNE_MODEL_PATH)


y_pred_t = fine_tune_model.predict(X_test_t)
rmse_t = np.sqrt(mean_squared_error(y_test_t, y_pred_t))
mae_t = mean_absolute_error(y_test_t, y_pred_t)
r2_t = r2_score(y_test_t, y_pred_t)
pearson_r_t, _ = pearsonr(y_test_t, y_pred_t)

print(f"RMSE: {rmse_t:.4f}, MAE: {mae_t:.4f}, R²: {r2_t:.4f}, Pearson R: {pearson_r_t:.4f}")

plot_data_t = pd.DataFrame({
    "True_periodic": y_test_t.values,
    "Pred_periodic": y_pred_t
})
plot_data_t.to_csv(f"plot_data_{TISSUE_NAME}.csv", index=False)

plt.figure(figsize=(8,8))
plt.scatter(y_test_t, y_pred_t, alpha=0.5)
plt.plot([y_single.min(), y_single.max()], [y_single.min(), y_single.max()], 'r--')
plt.xlabel("True periodicity")
plt.ylabel("Predicted periodicity")
plt.title(f"{TISSUE_NAME.capitalize()} Fine-tuned CatBoost (R={pearson_r_t:.3f}, R²={r2_t:.3f})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"CatBoost_scatter_{TISSUE_NAME}.png", dpi=300)
plt.close()

