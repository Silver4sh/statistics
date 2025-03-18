import os
import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def create_bulk_data():
    curr_dir = os.getcwd()
    file_path = os.path.join(curr_dir, 'data.csv')
    with open(file_path, 'w', newline='') as csv_file:
        fieldnames = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        writer = csv.writer(csv_file)
        writer.writerow(fieldnames)
    print('-' * 60)
    print('Isi nilai kuisioner di dalam file yang telah dibuat.')
    print('Direktori file berada di:', curr_dir)
    print('!!! FILE HARUS BERFORMAT CSV !!!')
    print('-' * 60)

def import_data():
    try:
        data = pd.read_csv('data.csv')
    except FileNotFoundError:
        raise FileNotFoundError("File 'data.csv' tidak ditemukan. Silakan jalankan create_bulk_data() terlebih dahulu.")
    data = data.dropna(how='all', axis=1)
    return data

def cronbach_alpha(data: pd.DataFrame) -> float:
    k = data.shape[1]
    if k < 2:
        raise ValueError("Data harus memiliki setidaknya dua item untuk menghitung Cronbach's alpha.")
    item_variances = data.var(ddof=1)
    total_score = data.sum(axis=1)
    total_variance = total_score.var(ddof=1)
    if total_variance == 0:
        raise ValueError("Variansi total adalah 0, periksa data Anda.")
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha

def item_total_correlation(data: pd.DataFrame) -> dict:
    total_score = data.sum(axis=1)
    correlations = {}
    for col in data.columns:
        correlations[col] = data[col].corr(total_score)
    return correlations

def linear_regression(data: pd.DataFrame) -> np.ndarray:
    data_copy = data.copy()
    data_copy['y'] = data_copy.sum(axis=1)
    X = data_copy.drop('y', axis=1).values
    y = data_copy['y'].values
    reg = LinearRegression().fit(X, y)
    return reg.coef_

def main():
    try:
        data = import_data()
    except FileNotFoundError as e:
        print(e)
        return
    
    try:
        alpha = cronbach_alpha(data)
        print("Reliabilitas (Cronbach's Alpha):", alpha)
    except Exception as e:
        print("Gagal menghitung reliabilitas:", e)
    
    try:
        correlations = item_total_correlation(data)
        print("Validitas (Item-Total Correlations):")
        for item, corr in correlations.items():
            print(f"  {item}: {corr}")
    except Exception as e:
        print("Gagal menghitung validitas:", e)
    
    try:
        coefs = linear_regression(data)
        print("Koefisien Regresi:", coefs)
    except Exception as e:
        print("Gagal melakukan regresi:", e)
    
    print("\nJika regresi gagal, mungkin data-mu juga lagi butuh liburan ;)")

if __name__ == "__main__":
    main()
