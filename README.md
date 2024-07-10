# Housing Price Prediction Project

## Project Overview

This project aims to predict the median house value for various districts in California using data from the 1990 census. The dataset contains one instance per district block group, with features such as geographical coordinates, median house age, total rooms, total bedrooms, population, households, median income, and proximity to the ocean. The goal is to design a regression model that accurately predicts the median house value based on these input attributes.

## Project Structure

- **Jupyter Notebook**: Contains all the code and analysis for data loading, preprocessing, visualization, splitting, scaling, modeling, and evaluation.
- **project_questions_and_solutions.pdf**: A PDF document answering non-coding questions and discussing the analysis.
- **Dataset**: California housing dataset (loaded directly using pandas).

## Dataset Description

- **longitude**: Geographical coordinate (continuous).
- **latitude**: Geographical coordinate (continuous).
- **housing_median_age**: Average age of houses in the district block (continuous).
- **total_rooms**: Total number of rooms in all houses in the district block (continuous).
- **total_bedrooms**: Total number of bedrooms in all houses in the district block (continuous).
- **population**: Number of people residing in the district block (continuous).
- **households**: Number of families in the district block (continuous).
- **median_income**: Median income for households in the district block (continuous, measured in tens of thousands of USD).
- **ocean_proximity**: Location of the house (categorical, e.g., inland, near the bay, near the ocean).
- **median_house_value**: Median house value within the district block (continuous, measured in USD).

## Project Steps

### Data Loading and Preprocessing

1. **Loading**: Load the dataset using `pandas.read_csv()` and store it in a DataFrame named `df`. Drop rows with missing values to ensure data integrity.
2. **Correlation Analysis**: Create a correlation DataFrame `corr_df` by dropping non-numerical columns and compute the Pearson correlation of each feature with the target variable `median_house_value`.
3. **Feature and Target Separation**: Separate features and target variable into DataFrames `X` and `Y` respectively.

### Data Visualization

1. **Histograms**: Plot histograms for key features to visualize their distributions.
2. **Descriptive Statistics**: Use `pandas.DataFrame.describe()` to report mean, median, and standard deviations.
3. **One-Hot Encoding**: Convert categorical variables into dummy/one-hot encoding using `pandas.get_dummies()`.

### Data Splitting

1. **Train-Test Split**: Split data into training and test sets using `sklearn.train_test_split()` with a 70-30 distribution.

### Data Scaling

1. **Standard Scaling**: Use `StandardScaler()` to scale the features and target values for both training and testing datasets.

### Modeling

1. **Linear Regression**: Fit a linear regression model using the scaled training data.
2. **Prediction**: Predict on the test data and transform predictions back to the original scale.

### Evaluation

1. **Performance Metrics**: Calculate MAPE, RMSE, and R² to evaluate model performance.
2. **Scatter Plot**: Plot a scatter plot comparing predicted and actual median house values.
3. **Feature Influence**: Rank continuous features based on their influence on the predictions.

### Principal Component Analysis (PCA)

1. **PCA Transformation**: Perform PCA on the features and plot a scatter plot of the first two principal components.
2. **Variance Explanation**: Report the total variance captured by the PCA components and the strength of each component.


## Conclusion

This project demonstrates the application of machine learning techniques to predict housing prices based on various features. By following the steps outlined above, you can reproduce the analysis and explore the model's performance.

## Author

Sowmya Buruju
