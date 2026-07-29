# Retail Sales Data Analysis and Business Analytics

## Project Overview

This project performs end-to-end data analysis on a retail sales dataset using Python. The goal is to analyze customer purchasing behavior, identify sales trends, and generate meaningful business insights through data preprocessing, feature engineering, and data visualization.

## Dataset

**Dataset Name:** Retail Sales Dataset

**Source:** https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset

**Note:** The dataset is not included in this repository. Please download the dataset from the above link and place the CSV file inside the `dataset/` folder as:

```
dataset/retail_sales.csv
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Project Workflow

- Loaded and explored the retail sales dataset
- Checked for missing values and duplicate records
- Converted the Date column into datetime format
- Performed feature engineering by creating:
  - Year
  - Month
  - Month Name
  - Day
  - Day of Week
- Conducted Exploratory Data Analysis (EDA)
- Generated business insights using visualizations

## Visualizations

The project includes the following visualizations:

- Gender Distribution
- Product Category Distribution
- Age Distribution
- Monthly Sales Trend
- Revenue by Product Category
- Quantity Sold by Category
- Correlation Heatmap

All generated graphs are available in the `images/` folder.

## Project Structure

```
Retail-Sales-Data-Analysis-and-Business-Analytics/
│
├── dataset/
├── images/
├── analysis.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

1. Clone this repository.
2. Download the dataset from Kaggle using the link above.
3. Place `retail_sales.csv` inside the `dataset/` folder.
4. Install the required libraries:

```bash
pip install -r requirements.txt
```

5. Run the project:

```bash
python analysis.py
```

## Key Insights

- Analyzed customer demographics and purchasing behavior.
- Identified sales trends across different product categories.
- Compared revenue and quantity sold by category.
- Visualized monthly sales performance and feature correlations.

## Author

**Deekshitha V**
