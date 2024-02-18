Vaghasiya, Om, 22205283

**Title - Laptop Price Prediction**

# Project Description

A Machine Learning Model to predict laptop price based on user requirement. This project uses Random Forest Regression Model to train on a given dataset. Interactive GUI is developed using PyQt6.

A file import is to be done using the file menu from the local device named laptopsdata.csv and after that the user should enter the specifications in the app and click on Get Recommendation button and the price is generated under the under with the other metrices such as MSE, RMSE, R-squared value and MAE is seen. Scatter plot, histogram , pairplot and heatmap are generated for data and user visualizations. Price is generated in Rupees.

# Installation

1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (Linux/Mac) or `.\venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Follow the steps in the Usage section.

# Basic Usage

Download this git repository or clone it to your system using following command:

`git clone https://mygit.th-deg.de/assistance-systems1/laptop-price-prediction`

Install required python packages from requirements.txt file using following command:

`pip install -r requirements.txt`

1. Run the app: `python src/main.py`
2. Load data using the file menu button and import data(Use the same CSV file provided here laptopsdata.csv).
3. Explore laptop data in the displayed table.
4. Enter the specifications of the laptop and click on Get Recommendation button
5. If you want to see data info you can again click on file menu and click on show data info and you can see the details and Heatmap.

# Implementation of the Requests

1. Graphical User Interface (GUI) Components

- **Function:** main window, buttons, QcomboBox, Qslider, QcheckBox and QTextBrowser for the input sections.
- Designed and implemented various UI components for improved aesthetics and usability.

2. Importing Data

- **Method:** `import_data`
- **Function:** Loads CSV data using `QFileDialog`.
- Improved file import functionality.

3. Preprocessing Data and Training the Model

- **Methods:** `preprocess_data`, `train_model`
- **Functions:** One-hot encodes categorical columns, scales numerical features, and trains the model using RandomForestRegressor.
- Optimized data preprocessing and model training.

4. Making price Recommendations

- **Method:** `predict_recommendation`
- **Function:** Gathers user inputs, constructs input data, and predicts laptop prices.
- Designed and connected user-friendly UI elements.

5. Calculating Metrics

- **Methods:** `calculate_mse`, `calculate_rmse`, `calculate_r_squared`, `calculate_mae`
- **Functions:** Compute regression metrics for model evaluation.
- Collaborated on accurate metric calculations.

6. Data Visualization and Graphs

- **Methods:** `show_data_info`, `plot_histogram`, `plot_pairplot`, `plot_histogram`, `plot_pairplot`, `plot_scatter`
- **Functions:** Generate informative visualizations.
- Designed visually appealing and informative plots.

# Work

Om Vaghasiya

1. Worked on Collection and Preprocessing of Dataset using Pandas.
2. Modifying and Preparing Data with use of numpy.
3. Training RandomForestRegression Model with help of Scikit-learn.
4. Data Visualization(matplotlib and pandas)
5. Python Programming
6. Creating a GUI Interface using PyQt6.
7. Getting User-Inputs using different GUI elements.
8. Integrating RandomForest Regression model to main file in order to get prediction price.
