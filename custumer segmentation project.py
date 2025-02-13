import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
try:
    data = pd.read_csv(r"C:\Users\msi-pc\Downloads\Telegram Desktop\e_commerce_data.csv")
except FileNotFoundError:
    print("Error: The specified file was not found.")
    exit()

# Data Cleaning
## Drop rows with missing values in critical columns
data.dropna(subset=['Customer_ID', 'Transaction_ID', 'Product_Name', 'Quantity', 'Unit_Price', 'Transaction_Date'], inplace=True)

## Fill missing values in non-critical columns
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Gender'].fillna('Unknown', inplace=True)
data['Region'].fillna('Unknown', inplace=True)
data['Payment_Method'].fillna('Unknown', inplace=True)
data['Unit_Cost'].fillna(0, inplace=True)

## Ensure correct data types
data['Quantity'] = data['Quantity'].astype(int)
data['Unit_Price'] = data['Unit_Price'].astype(float)
data['Unit_Cost'] = data['Unit_Cost'].astype(float)
data['Transaction_Date'] = pd.to_datetime(data['Transaction_Date'], errors='coerce')

## Sort transaction dates in chronological order
data.sort_values(by='Transaction_Date', inplace=True)

# Customer Analysis
## Number of transactions for each customer
number_of_transactions = data.groupby('Customer_ID').size().reset_index(name='Transaction Number')

## Identify high-frequency buyers and their total spending
customer_summary = data.groupby('Customer_ID').agg(
    Total_spending=('Unit_Price', 'sum'),
    Transaction_count=('Transaction_ID', 'count')
).reset_index()

high_frequency_threshold = 5
customer_summary['High_Frequency_Buyer'] = customer_summary['Transaction_count'] > high_frequency_threshold

## Segment customers by spending habits
spending_bins = [0, 100, 500, 1000, 5000, np.inf]
spending_labels = ['Low', 'Medium', 'High', 'Very High', 'Top']
customer_summary['Spending_segment'] = pd.cut(customer_summary['Total_spending'], bins=spending_bins, labels=spending_labels)

## Merge customer summary back into the original data
data = data.merge(customer_summary[['Customer_ID', 'Total_spending', 'Spending_segment']], on='Customer_ID', how='left')

## Segment customers by demographics
age_bins = [18, 25, 35, 45, 55, 65, np.inf]
age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
data['Age_segment'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)

# Product Analysis
## Calculate revenue and profitability
data['Revenue'] = data['Quantity'] * data['Unit_Price']
data['Profit'] = data['Unit_Price'] - data['Unit_Cost']

## Summarize product performance
product_summary = data.groupby('Product_Name').agg(
    Total_Quantity=('Quantity', 'sum'),
    Total_Revenue=('Revenue', 'sum'),
    Total_Profit=('Profit', 'sum')
).sort_values(by='Total_Quantity', ascending=False)

top_selling_by_quantity = product_summary.sort_values(by='Total_Quantity', ascending=False)
top_profitable_products = product_summary.sort_values(by='Total_Profit', ascending=False).head(10)

# Time Series Analysis
## Extract year, month, and day from the transaction date
data['Year'] = data['Transaction_Date'].dt.year
data['Month'] = data['Transaction_Date'].dt.month

## Identify sales trends over time
sales_trends = data.groupby(['Year', 'Month']).agg(Total_Revenue=('Revenue', 'sum')).reset_index()

# Plotting Functions
def plot_sales_trends(sales_trends):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=sales_trends, x='Month', y='Total_Revenue', hue='Year', marker='o')
    plt.title('Monthly Sales Trends Over Years')
    plt.xlabel('Month')
    plt.ylabel('Total Revenue')
    plt.legend(title='Year')
    plt.show()

def plot_bar(data, x, y, title, xlabel, ylabel, rotation=0, hue=None):
    plt.figure(figsize=(14, 7))
    sns.barplot(data=data, x=x, y=y, palette='viridis', hue=hue)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.show()

# Plot sales trends
plot_sales_trends(sales_trends)

## Seasonal Sales Trends
data['Season'] = data['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                             'Spring' if x in [3, 4, 5] else
                                             'Summer' if x in [6, 7, 8] else
                                             'Fall')

seasonal_sales = data.groupby('Season').agg(Total_Revenue=('Revenue', 'sum')).reset_index()
plot_bar(seasonal_sales, x='Season', y='Total_Revenue', title='Seasonal Sales Trends', xlabel='Season', ylabel='Total Revenue')

# Regional Sales Performance
region_sales = data.groupby('Region').agg(Total_Revenue=('Revenue', 'sum')).reset_index()
plot_bar(region_sales, x='Region', y='Total_Revenue', title='Sales Performance by Region', xlabel='Region', ylabel='Total Revenue', rotation=45)

# Analyze regional preferences for product categories
region_category_preference = data.groupby(['Region', 'Product_Category']).agg(Total_Quantity=('Quantity', 'sum')).reset_index()
plot_bar(region_category_preference, x='Region', y='Total_Quantity', title='Region Preferences for Product Categories', xlabel='Region', ylabel='Total Quantity', rotation=45, hue='Product_Category')

# Analyze payment method preferences
payment_method_summary = data.groupby('Payment_Method').agg(Count=('Payment_Method', 'count')).reset_index()
plot_bar(payment_method_summary, x='Payment_Method', y='Count', title='Most Commonly Used Payment Methods', xlabel='Payment Method', ylabel='Count', rotation=45)

# Analyze high-value transactions
high_value_threshold = 40
high_value_transactions = data[data['Total_spending'] > high_value_threshold]
payment_method_high_value = high_value_transactions.groupby('Payment_Method').agg(Total_High_Value_Transactions=('Total_spending', 'count')).reset_index()
plot_bar(payment_method_high_value, x='Payment_Method', y='Total_High_Value_Transactions', title='High-Value Transactions by Payment Method', xlabel='Payment Method', ylabel='Total High-Value Transactions', rotation=45)

# Examine spending habits by age segment
age_spending = data.groupby('Age_segment').agg(Average_Spending=('Total_spending', 'mean')).reset_index()
plot_bar(age_spending, x='Age_segment', y='Average_Spending', title='Average Spending by Age Segment', xlabel='Age Segment', ylabel='Average Spending')

# Gender Analysis
gender_product_preference = data.groupby(['Gender', 'Product_Category']).agg(Total_Quantity=('Quantity', 'sum')).reset_index()
plot_bar(gender_product_preference, x='Gender', y='Total_Quantity', title='Gender Preferences for Product Categories', xlabel='Gender', ylabel='Total Quantity', hue='Product_Category')

# High-value customers
high_value_customers = customer_summary[customer_summary['Total_spending'] > customer_summary['Total_spending'].quantile(0.95)]
plot_bar(high_value_customers, x='Customer_ID', y='Total_spending', title='High-Value Customers', xlabel='Customer ID', ylabel='Total Spending', rotation=90)

# Outlier Analysis
quantity_outliers = data[(data['Quantity'] > data['Quantity'].quantile(0.99)) | (data['Quantity'] < data['Quantity'].quantile(0.01))]
unit_price_outliers = data[(data['Unit_Price'] > data['Unit_Price'].quantile(0.99)) | (data['Unit_Price'] < data['Unit_Price'].quantile(0.01))]

## Plot outliers in quantity
plt.figure(figsize=(10, 5))
plt.scatter(data.index, data['Quantity'], alpha=0.5)
plt.scatter(quantity_outliers.index, quantity_outliers['Quantity'], color='red')
plt.title('Outliers in Quantity')
plt.xlabel('Index')
plt.ylabel('Quantity')
plt.show()

## Plot outliers in unit price
plt.figure(figsize=(10, 5))
plt.scatter(data.index, data['Unit_Price'], alpha=0.5)
plt.scatter(unit_price_outliers.index, unit_price_outliers['Unit_Price'], color='red')
plt.title('Outliers in Unit Price')
plt.xlabel('Index')
plt.ylabel('Unit Price')
plt.show()

# Heatmap for regional demand for products
region_product_summary = data.groupby(['Region', 'Product_Name']).agg(Total_Quantity=('Quantity', 'sum')).reset_index()
region_product_pivot = region_product_summary.pivot(index='Region', columns='Product_Name', values='Total_Quantity').fillna(0)

plt.figure(figsize=(14, 7))
sns.heatmap(region_product_pivot, cmap='YlGnBu', annot=True, fmt='g')
plt.title('Regional Demand for Products')
plt.xlabel('Product Name')
plt.ylabel('Region')
plt.xticks(rotation=90)
plt.show()