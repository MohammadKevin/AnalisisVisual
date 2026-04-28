import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data_praktikum_analisis_data (1) - data_praktikum_analisis_data (1).csv')

df.columns = df.columns.str.strip().str.lower()

print("Kolom dalam dataset:")
print(df.columns)

print("\nPreview data:")
print(df.head())

kolom = {
    "price": None,
    "order_date": None,
    "total_sales": None,
    "ad_budget": None,
    "discount": None,
    "product": None
}

for col in df.columns:
    if "price" in col:
        kolom["price"] = col
    elif "date" in col:
        kolom["order_date"] = col
    elif "sales" in col or "revenue" in col:
        kolom["total_sales"] = col
    elif "ad" in col or "budget" in col:
        kolom["ad_budget"] = col
    elif "discount" in col:
        kolom["discount"] = col
    elif "product" in col or "name" in col:
        kolom["product"] = col

print("\nMapping Kolom:")
print(kolom)

if kolom["price"]:
    df = df[df[kolom["price"]] > 0]

if kolom["order_date"]:
    df[kolom["order_date"]] = pd.to_datetime(df[kolom["order_date"]])

df = df.dropna()

if kolom["order_date"]:
    df['month'] = df[kolom["order_date"]].dt.to_period('M').astype(str)

if kolom["total_sales"] and "month" in df.columns:
    monthly_sales = df.groupby('month')[kolom["total_sales"]].sum()

    plt.figure(figsize=(10,5))
    plt.plot(monthly_sales.index, monthly_sales.values, marker='o')
    plt.title('Tren Penjualan Bulanan')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

cols_corr = [kolom["total_sales"], kolom["ad_budget"], kolom["discount"]]
cols_corr = [c for c in cols_corr if c is not None]

if len(cols_corr) >= 2:
    correlation = df[cols_corr].corr()

    plt.figure(figsize=(6,4))
    sns.heatmap(correlation, annot=True)
    plt.title('Peta Korelasi')
    plt.show()

if kolom["product"] and kolom["total_sales"]:
    top_products = df.groupby(kolom["product"])[kolom["total_sales"]] \
.sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10,5))
    top_products.plot(kind='bar')
    plt.title('Top 10 Produk Terlaris')
    plt.xticks(rotation=45)
    plt.show()

print("\nInsight:")
print("- Lihat grafik tren untuk pola penjualan")
print("- Gunakan korelasi untuk cek pengaruh iklan & diskon")
print("- Fokus pada produk terlaris untuk strategi marketing")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if kolom["ad_budget"] and kolom["total_sales"]:
    
    # Ambil data
    X = df[[kolom["ad_budget"]]]
    y = df[kolom["total_sales"]]

    # Split data training & testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Buat model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluasi
    r2 = model.score(X_test, y_test)
    coef = model.coef_[0]

    print("\nRegresi Linear:")
    print(f"Koefisien Iklan: {coef}")
    print(f"Akurasi Model (R2 Score): {r2}")

    # Visualisasi garis regresi
    plt.figure(figsize=(8,5))
    plt.scatter(X, y)
    plt.plot(X, model.predict(X))
    plt.xlabel("Ad Budget")
    plt.ylabel("Total Sales")
    plt.title("Regresi Linear: Iklan vs Penjualan")
    plt.show()