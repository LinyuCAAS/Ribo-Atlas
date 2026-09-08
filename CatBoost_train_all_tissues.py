import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import os
import shap  

DATA_PATH = "Features.csv"
GLOBAL_MODEL_PATH = "Multi_Tissues_CatBoost_uORF_model.cbm"

df = pd.read_csv(DATA_PATH)

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

df["tissue"] = df["tissue"].astype("category")

X = df.drop(["periodic"], axis=1)
y = df["periodic"]
cat_features = ["tissue"]

X_temp, X_final_test, y_temp, y_final_test = train_test_split(
    X, y, test_size=0.05, random_state=42
)



X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)


global_model = CatBoostRegressor(
    iterations=20000,
    depth=6,
    learning_rate=0.03,
    loss_function='RMSE',
    eval_metric='R2',
    random_seed=42,
    od_type='Iter',
    od_wait=20,
    thread_count=8,
    verbose=100
)

global_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val), 
    cat_features=cat_features
)


global_model.save_model(GLOBAL_MODEL_PATH)


y_pred_final = global_model.predict(X_final_test)
rmse = np.sqrt(mean_squared_error(y_final_test, y_pred_final))
mae = mean_absolute_error(y_final_test, y_pred_final)
r2 = r2_score(y_final_test, y_pred_final)
pearson_r, _ = pearsonr(y_final_test, y_pred_final)

print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Pearson R: {pearson_r:.4f}")

plot_data = pd.DataFrame({
    "True_periodic": y_final_test.values,
    "Pred_periodic": y_pred_final
})
plot_data.to_csv("plot_data_final_test.csv", index=False)

importances = global_model.get_feature_importance()
imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)
imp_df.to_csv("feature_importance_global.csv", index=False)

plt.figure(figsize=(8,8))
plt.scatter(y_final_test, y_pred_final, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("True periodicity")
plt.ylabel("Predicted periodicity")
plt.title(f"Global CatBoost (Final Test | R={pearson_r:.3f}, R²={r2:.3f})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("CatBoost_scatter_final_test.png", dpi=300)
plt.close()


explainer = shap.Explainer(global_model)
shap_values = explainer(X_val)

shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
shap_df.to_csv("shap_values.csv", index=False)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_val, show=False)
plt.tight_layout()
plt.savefig("SHAP_summary_plot.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 8))
shap.plots.heatmap(shap_values, show=False)
plt.tight_layout()
plt.savefig("SHAP_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

top_feat = imp_df.iloc[0]["Feature"]
plt.figure(figsize=(8, 6))
shap.dependence_plot(top_feat, shap_values.values, X_val, show=False)
plt.tight_layout()
plt.savefig(f"SHAP_dependence_{top_feat}.png", dpi=300, bbox_inches='tight')
plt.close()

