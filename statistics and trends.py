"""
Statistics and trends: fuel economy of 398 cars, model years 1970-1982
(Kaggle "Auto-mpg dataset", originally UCI / CMU StatLib).

Produces three figures for the report:
    categorical_plot.png  - histogram of mpg, stacked by region of origin
    relational_plot.png   - scatter of fuel consumption (L/100 km) against weight
    statistical_plot.png  - box plot of mpg by model year

Usage:
    python statistics_and_trends.py [path/to/auto-mpg.csv]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
ORIGINS = ['USA', 'Europe', 'Japan']
COLOURS = {'USA': BLUE, 'Europe': ORANGE, 'Japan': AQUA}
MARKERS = {'USA': 'o', 'Europe': '^', 'Japan': 's'}
L100KM = 235.215          # litres/100 km = 235.215 / (US) miles per gallon


def set_style():
    """Shared, quiet plotting style sized for a two-page A4 report."""
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 300, 'figure.dpi': 120,
    })


def plot_categorical_plot(df):
    """Histogram of mpg in 2 mpg bins, stacked by region of origin."""
    fig, ax = plt.subplots(figsize=(6.85, 2.4))
    bins = np.arange(8, 49, 2)
    data = [df.loc[df['origin'] == o, 'mpg'] for o in ORIGINS]
    ax.hist(data, bins=bins, stacked=True, color=[COLOURS[o] for o in ORIGINS],
            edgecolor='white', linewidth=0.8,
            label=[f'{o} (n={len(x)}, median {x.median():.1f})'
                   for o, x in zip(ORIGINS, data)])
    top = np.histogram(df['mpg'], bins)[0].max()
    mean, med = df['mpg'].mean(), df['mpg'].median()
    ax.axvline(med, color=INK, linestyle='--', linewidth=1)
    ax.axvline(mean, color=INK, linestyle=':', linewidth=1.2)
    ax.text(med - 0.3, top * 1.12, f'median\n{med:.1f}', ha='right', va='top', fontsize=6.8)
    ax.text(mean + 0.3, top * 1.12, f'mean\n{mean:.2f}', ha='left', va='top', fontsize=6.8)
    ax.set_ylim(0, top * 1.15)
    ax.set_xlim(8, 48)
    ax.set_xlabel('Fuel economy (miles per US gallon, 2 mpg bins)')
    ax.set_ylabel('Number of cars')
    ax.set_title('Distribution of fuel economy, all model years', loc='left')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig('categorical_plot.png')
    plt.close(fig)


def plot_relational_plot(df):
    """Scatter of fuel consumption against weight, with a straight-line fit."""
    fig, ax = plt.subplots(figsize=(6.85, 2.6))
    for o in ORIGINS:
        part = df[df['origin'] == o]
        ax.scatter(part['weight'], part['l_per_100km'], s=13, marker=MARKERS[o],
                   c=COLOURS[o], alpha=0.75, edgecolors='white', linewidths=0.4,
                   label=o)
    slope, intercept = np.polyfit(df['weight'], df['l_per_100km'], 1)
    xs = np.array([1550, 5200])
    ax.plot(xs, slope * xs + intercept, color=INK2, linestyle='--', linewidth=1.3,
            label=f'Fit: +{slope * 1000:.1f} L/100 km per 1,000 lb')
    ax.set_xlabel('Weight (lb)')
    ax.set_ylabel('Fuel consumption (L/100 km)')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_title('Fuel consumption against weight (L/100 km = 235.2 / mpg)', loc='left')
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig('relational_plot.png')
    plt.close(fig)


def plot_statistical_plot(df):
    """Box plot of mpg for each model year, with the number of cars per year."""
    years = sorted(df['model_year'].unique())
    data = [df.loc[df['model_year'] == y, 'mpg'] for y in years]
    fig, ax = plt.subplots(figsize=(6.85, 3.0))
    box = ax.boxplot(data, patch_artist=True, widths=0.6,
                     medianprops=dict(color='white', linewidth=1.6),
                     whiskerprops=dict(color=INK2, linewidth=0.9),
                     capprops=dict(color=INK2, linewidth=0.9),
                     flierprops=dict(marker='o', markersize=2.5, alpha=0.6,
                                     markerfacecolor=INK2, markeredgewidth=0))
    for patch in box['boxes']:
        patch.set_facecolor(BLUE)
        patch.set_edgecolor(BLUE)
    medians = [x.median() for x in data]
    ax.plot(range(1, len(years) + 1), medians, color=ORANGE, linewidth=1.4,
            marker='o', markersize=3, label='Median mpg', zorder=3)
    ax.set_xticks(range(1, len(years) + 1))
    ax.set_xticklabels([f'19{y}\nn={len(x)}' for y, x in zip(years, data)])
    for year, label in [(73, '1973 oil crisis'), (79, '1979 oil crisis')]:
        i = years.index(year) + 1
        ax.annotate(label, xy=(i, 44), xytext=(i, 44), ha='center', fontsize=6.8,
                    color=INK2)
    ax.set_ylim(5, 48)
    ax.set_ylabel('Miles per gallon')
    ax.set_title('Fuel economy by model year', loc='left')
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig('statistical_plot.png')
    plt.close(fig)


def statistical_analysis(df, col):
    """Return the first four moments of a column (excess kurtosis)."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Tidy labels, report missing horsepower and add fuel consumption."""
    df = df.drop_duplicates().copy()
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    print('missing horsepower:', df['horsepower'].isna().sum())
    if pd.api.types.is_numeric_dtype(df['origin']):         # Kaggle file codes 1/2/3
        df['origin'] = df['origin'].map({1: 'usa', 2: 'europe', 3: 'japan'})
    df['origin'] = df['origin'].map({'usa': 'USA', 'europe': 'Europe', 'japan': 'Japan'})
    if 'model year' in df.columns:                         # Kaggle column name
        df = df.rename(columns={'model year': 'model_year', 'car name': 'name'})
    df['l_per_100km'] = L100KM / df['mpg']
    print(df.describe().round(2))
    print(df[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model_year']].corr().round(3)['mpg'])
    return df


def writing(moments, col):
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    skew = 'right' if moments[2] > 0.5 else 'left' if moments[2] < -0.5 else 'not'
    kurt = ('leptokurtic' if moments[3] > 1 else
            'platykurtic' if moments[3] < -1 else 'mesokurtic')
    print(f'The data was {skew} skewed and {kurt}.')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'auto-mpg.csv'
    set_style()
    df = preprocessing(pd.read_csv(path))
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    writing(statistical_analysis(df, 'mpg'), 'mpg')
    print(f"r(weight, mpg) = {ss.pearsonr(df['weight'], df['mpg'])[0]:.3f}, "
          f"r(weight, L/100km) = {ss.pearsonr(df['weight'], df['l_per_100km'])[0]:.3f}")
    X = np.c_[np.ones(len(df)), df['weight'] / 1000, df['model_year']]
    coef = np.linalg.lstsq(X, df['mpg'].to_numpy(), rcond=None)[0]
    print(f'mpg ~ weight + year: {coef[1]:.2f} mpg per 1,000 lb, '
          f'{coef[2]:+.2f} mpg per model year')
    print(df.groupby('model_year')['mpg'].median())


if __name__ == '__main__':
    main()
