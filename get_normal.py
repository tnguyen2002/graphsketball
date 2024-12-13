import os
import json
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Directories containing the data
player_data_dir = 'data/player_season_jsons/'
standings_data_dir = 'data/standings_jsons/'

def get_win_percentages(standings_data_dir):
    """
    Reads the standings data from all seasons and computes the win percentages for each team.

    Returns:
        A list of all team win percentages across all seasons.
    """
    seasons = [f"{year - 1}_{year}" for year in range(2000, 2025)]
    win_percentages = []

    for season in seasons:
        standings_file = os.path.join(standings_data_dir, f'{season}_standings.json')

        if not os.path.exists(standings_file):
            continue

        # Load standings data
        with open(standings_file, 'r') as f:
            standings_data = json.load(f)

        # Compute win percentages for the season
        for team_data in standings_data:
            wins = team_data['wins']
            losses = team_data['losses']
            win_percentage = wins / (wins + losses)
            win_percentages.append(win_percentage)

    return win_percentages

def fit_normal_distribution(win_percentages):
    """
    Fits a normal distribution to the given win percentages and plots the results.

    Args:
        win_percentages: A list of win percentages to fit the distribution to.

    Returns:
        mean: The mean of the fitted normal distribution.
        std: The standard deviation of the fitted normal distribution.
    """
    # Fit a normal distribution to the data
    mean, std = norm.fit(win_percentages)

    # Plot the histogram and the fitted distribution
    plt.figure(figsize=(10, 6))
    plt.hist(win_percentages, bins=20, density=True, alpha=0.6, color='g', label='Histogram of Win Percentages')

    # Plot the PDF of the fitted normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Normal Fit: $\mu={mean:.2f}, \sigma={std:.2f}$')

    plt.title('Win Percentage Distribution and Fitted Normal Distribution')
    plt.xlabel('Win Percentage')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    return mean, std

if __name__ == "__main__":
    # Step 1: Get win percentages
    win_percentages = get_win_percentages(standings_data_dir)

    # Step 2: Fit normal distribution and plot results
    mean, std = fit_normal_distribution(win_percentages)

    print(f"Fitted Normal Distribution: Mean = {mean:.4f}, Std Dev = {std:.4f}")