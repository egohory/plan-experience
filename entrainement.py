import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from statsmodels.formula.api import ols

'''
    P : Numéro d'expérience
    D : Jour d'entrainement, mardi ou jeudi (1 ou -1)
    W : Météo, beau temps ou pluie (1 ou -1)
    T : Température pendant l'entrainement, chaud (>10°C) ou froid (<10°C) (1 ou -1)
    G : Semaine de match ? Match le dimanche ou repos le dimanche (1 ou -1)
    Presence : Moyenne des joueurs présents à l'entrainement sur ces jours-là
'''
# Lecture du fichier Excel contenant les données de présence
df = pd.read_excel("releves_presence.xlsx")

print(df)
# Définition de la formule du modèle linéaire avec les interactions
formula = 'Presence ~ D + W + T + G + D:W + D:T + D:G + W:T + W:G + T:G'

# Fit model
model = ols(formula, data=df).fit()


# Creation du graph 3D
def plot_3d_predictions(model, df, factor_x, factor_y, response_var, num_points=50):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of the measured data
    ax.scatter(df[factor_x], df[factor_y], df[response_var], color='blue', label='Measured Data', s=50)

    # Generate a grid of values for Temperature and Stirring_rate
    x_range = np.linspace(df[factor_x].min(), df[factor_x].max(), num_points)
    y_range = np.linspace(df[factor_y].min(), df[factor_y].max(), num_points)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Create a DataFrame for the grid and set other factors to their mean values
    grid_data = pd.DataFrame({factor_x: X.ravel(), factor_y: Y.ravel()})
    for col in df.columns:
        if col not in [factor_x, factor_y, response_var]:
            grid_data[col] = df[col].mean()
    
    # Predict filtration rate on the grid using the model
    Z = model.predict(grid_data).values.reshape(num_points, num_points)
    
    # Plot the predicted surface
    ax.plot_surface(X, Y, Z, color='lightgreen', alpha=0.6, edgecolor='k', label='Predicted Surface')
    
    # Labels and legend
    ax.set_xlabel(factor_x)
    ax.set_ylabel(factor_y)
    ax.set_zlabel(response_var)
    ax.set_title(f"Représentation 3D de {response_var} vs {factor_x} et {factor_y}")
    ax.legend()
    plt.savefig(f"plots/3d_plot_{factor_x}_{factor_y}.jpeg", dpi=600)

# Appel de la fonction pour chaque paire de facteurs
factor_list = ['T', 'W', 'D', 'G']
for i, factor_x in enumerate(factor_list):
    for j, factor_y in enumerate(factor_list):
        if i < j:
            plot_3d_predictions(model, df, factor_x, factor_y, 'Presence')