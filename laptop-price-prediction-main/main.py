import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QWidget,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QTextBrowser,
    QSlider,
    QCheckBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class LaptopPricePredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Laptop Price Prediction")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Add a toolbar
        self.toolbar = self.addToolBar("Toolbar")

        # Create a File menu and add actions
        file_menu = self.menuBar().addMenu("File")

        import_action = QAction("Import Data", self)
        import_action.triggered.connect(self.import_data)
        import_action.setShortcut(QKeySequence.StandardKey.Open)  # Use Open shortcut
        file_menu.addAction(import_action)

        show_info_action = QAction("Show Data Info", self)
        show_info_action.triggered.connect(self.show_data_info)
        file_menu.addAction(show_info_action)

        # Feature input layout
        self.feature_layout = QFormLayout()
        self.Company_input = QLineEdit(self)
        self.feature_layout.addRow("Company:", self.Company_input)

        # CPU Speed Slider
        self.cpu_speed_slider = QSlider()
        self.cpu_speed_slider.setOrientation(Qt.Orientation.Horizontal)
        self.cpu_speed_slider.setRange(10, 30)
        self.cpu_speed_slider.setValue(15)
        self.cpu_speed_slider.valueChanged.connect(self.update_cpu_speed_label)

        self.cpu_speed_label = QLabel(
            f"CPU Speed: {self.cpu_speed_slider.value() / 10}", self
        )
        self.feature_layout.addRow(self.cpu_speed_label, self.cpu_speed_slider)

        # Screen Inches Slider
        self.screen_inches_slider = QSlider()
        self.screen_inches_slider.setOrientation(Qt.Orientation.Horizontal)
        self.screen_inches_slider.setRange(130, 180)
        self.screen_inches_slider.setValue(150)
        self.screen_inches_slider.valueChanged.connect(self.update_screen_inches_label)
        self.screen_inches_label = QLabel(
            f"Screen Inches: {self.screen_inches_slider.value() / 10}", self
        )
        self.feature_layout.addRow(self.screen_inches_label, self.screen_inches_slider)

        # Ram Size ComboBox
        self.ram_size_combobox = QComboBox(self)
        ram_sizes = ["4GB", "8GB", "16GB", "32GB", "64GB"]
        self.ram_size_combobox.addItems(ram_sizes)
        self.feature_layout.addRow("Ram Size:", self.ram_size_combobox)

        # Type Name ComboBox
        self.type_name_combobox = QComboBox(self)
        type_names = [
            "Gaming",
            "Notebook",
            "Ultrabook",
            "2 in 1 Convertible",
            "Workstation",
        ]
        self.type_name_combobox.addItems(type_names)
        self.feature_layout.addRow("Type Name:", self.type_name_combobox)

        # Checkboxes
        self.ipspanel_checkbox = QCheckBox("IPS Panel", self)
        self.feature_layout.addRow(self.ipspanel_checkbox)

        self.retinadisplay_checkbox = QCheckBox("Retina Display", self)
        self.feature_layout.addRow(self.retinadisplay_checkbox)

        # Storage checkboxes
        self.ssd_checkbox = QCheckBox("SSD", self)
        self.ssd_checkbox.stateChanged.connect(self.ssd_checkbox_state_changed)
        self.feature_layout.addRow(self.ssd_checkbox)

        # Storage Size ComboBox
        self.ssd_size_combobox = QComboBox(self)
        ssd_sizes = ["0GB", "128GB", "256GB", "512GB"]
        self.ssd_size_combobox.addItems(ssd_sizes)
        self.ssd_size_combobox.setEnabled(True)
        self.feature_layout.addRow("Storage Size:", self.ssd_size_combobox)

        # GPU Brand ComboBox
        self.gpu_brand_combobox = QComboBox(self)
        self.gpu_brand_combobox.addItems(["Nvidia", "AMD", "Intel"])
        self.feature_layout.addRow("GPU Brand:", self.gpu_brand_combobox)

        # OpSys ComboBox
        self.OpSys_combobox = QComboBox(self)
        OpSys_options = ["windows", "macos", "linux"]
        self.OpSys_combobox.addItems(OpSys_options)
        self.feature_layout.addRow("Operating System:", self.OpSys_combobox)

        # Laptop Weight Slider
        self.weight_slider = QSlider()
        self.weight_slider.setOrientation(Qt.Orientation.Horizontal)
        self.weight_slider.setRange(1, 50)
        self.weight_slider.setValue(25)
        self.weight_slider.valueChanged.connect(self.update_weight_label)

        self.weight_label = QLabel(
            f"Laptop Weight: {self.weight_slider.value() / 10} kg", self
        )
        self.feature_layout.addRow(self.weight_label, self.weight_slider)
        self.layout.addLayout(self.feature_layout)

        # Get Recommendation button
        self.predict_button = QPushButton("Get Recommendation", self)
        self.predict_button.clicked.connect(self.predict_recommendation)
        self.layout.addWidget(self.predict_button)

        # Metrics label
        self.metric_label = QLabel("Metrics:", self)
        self.layout.addWidget(self.metric_label)

        # Text Browser
        self.text_browser = QTextBrowser(self)
        self.layout.addWidget(self.text_browser)

        # Recommendation Canvas
        self.canvas = RecommendationCanvas(self)
        self.layout.addWidget(self.canvas)

        # Data variables
        self.df = None
        self.model = None
        self.scaler = None  # Added scaler attribute
        self.X_train = None  # Added X_train attribute

    def update_cpu_speed_label(self, value):
        cpu_speed_value = value / 10
        self.cpu_speed_label.setText(f"CPU Speed: {cpu_speed_value}")

    def update_screen_inches_label(self, value):
        screen_inches_value = value / 10
        self.screen_inches_label.setText(f"Screen Inches: {screen_inches_value}")

    def update_weight_label(self, value):
        weight_value = value / 10
        self.weight_label.setText(f"Laptop Weight: {weight_value} kg")

    def import_data(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)"
        )

        if file_path:
            self.df = pd.read_csv(file_path)

            # Fit and save the scaler
            self.scaler = StandardScaler()
            self.scaler.fit(self.df[["Inches", "cpu_speed", "Weight_kg"]])

            # Preprocess data and train the model
            self.preprocess_data()
            self.train_model()

    def preprocess_data(self):
        # Check if categorical columns are present in the DataFrame
        existing_categorical_columns = [
            col
            for col in self.df.columns
            if col in ["gpu_brand", "OpSys", "TypeName", "Ram"]
        ]

        # One-hot encode existing categorical columns
        self.df = pd.get_dummies(
            self.df,
            columns=existing_categorical_columns,
            prefix=existing_categorical_columns,
        )

        # Scale numerical features using the saved scaler
        numerical_columns = ["Inches", "cpu_speed", "Weight_kg"]
        self.df[numerical_columns] = self.scaler.transform(self.df[numerical_columns])

    def train_model(self):
        # Features and target variable
        features = [
            "Inches",
            "cpu_speed",
            "gpu_brand_AMD",
            "gpu_brand_Nvidia",
            "gpu_brand_Intel",
            "OpSys_windows",
            "OpSys_macos",
            "OpSys_linux",
            "ipspanel",
            "retinadisplay",
            "ssd",
            "Weight_kg",
            "Ram_4",
            "Ram_8",
            "Ram_16",
            "Ram_32",
            "Ram_64",
            "TypeName_Gaming",
            "TypeName_Notebook",
            "TypeName_Ultrabook",
            "TypeName_2 in 1 Convertible",
            "TypeName_Workstation",
        ]
        target = "Price"

        X = self.df[features]
        y = self.df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Use RandomForestRegressor instead of LinearRegression
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def predict_recommendation(self):
        if self.df is not None:
            Company = self.Company_input.text()
            cpu_speed = float(self.cpu_speed_slider.value()) / 10
            screen_inches = float(self.screen_inches_slider.value()) / 10
            gpu_brand = self.gpu_brand_combobox.currentText()
            ram_size_text = self.ram_size_combobox.currentText()
            ram_size = int(ram_size_text.replace("GB", ""))
            type_name = self.type_name_combobox.currentText()
            OpSys = self.OpSys_combobox.currentText()
            weight = self.weight_slider.value() / 10
            ipspanel = self.ipspanel_checkbox.isChecked()
            retinadisplay = self.retinadisplay_checkbox.isChecked()
            ssd = self.ssd_checkbox.isChecked()

            # Initialize input_data with user's specifications using NumPy arrays
            input_data = pd.DataFrame(
                {
                    "Inches": np.array([screen_inches]),
                    "cpu_speed": np.array([cpu_speed]),
                    "gpu_brand_AMD": np.array([0]),
                    "gpu_brand_Nvidia": np.array([0]),
                    "gpu_brand_Intel": np.array([0]),
                    "OpSys_windows": np.array([0]),
                    "OpSys_macos": np.array([0]),
                    "OpSys_linux": np.array([0]),
                    "ipspanel": np.array([int(ipspanel)]),
                    "retinadisplay": np.array([int(retinadisplay)]),
                    "ssd": np.array([int(ssd)]),
                    "Weight_kg": np.array([weight]),
                    "Ram_4": np.array([0]),
                    "Ram_8": np.array([0]),
                    "Ram_16": np.array([0]),
                    "Ram_32": np.array([0]),
                    "Ram_64": np.array([0]),
                    "TypeName_Gaming": np.array([0]),
                    "TypeName_Notebook": np.array([0]),
                    "TypeName_Ultrabook": np.array([0]),
                    "TypeName_2 in 1 Convertible": np.array([0]),
                    "TypeName_Workstation": np.array([0]),
                }
            )

            input_data["gpu_brand_" + gpu_brand] = np.array([1])
            input_data[f"OpSys_{OpSys.lower()}"] = np.array([1])
            input_data[f"Ram_{ram_size}"] = np.array([1])
            input_data[f"TypeName_{type_name}"] = np.array([1])
            input_data[f"gpu_brand_{gpu_brand}"] = np.array([1])

            if ssd:
                storage_size = self.ssd_size_combobox.currentText()
                input_data[f"storage_size_{storage_size}"] = np.array([1])

            # Duplicate the row for multiple predictions (adjust the number as needed)
            input_data = pd.concat([input_data] * 255, ignore_index=True)

            input_data = input_data[self.X_test.columns]

            self.y_test_pred = self.model.predict(input_data)

            self.metric_label.setText(
                f"Predicted Price: ₹{self.y_test_pred[0]:,.2f} | R-squared: {self.calculate_r_squared()} | MAE: {self.calculate_mae()}  | MSE: {self.calculate_mse()} | RMSE: {self.calculate_rmse()}"
            )
            self.canvas.plot_scatter(self.df, input_data, self.model)
            self.plot_histogram()
            self.plot_pairplot()

    def calculate_mse(self):
        y_test_pred = self.model.predict(self.X_train)
        mse = mean_squared_error(self.y_train, y_test_pred)
        return round(mse, 2)

    def calculate_rmse(self):
        y_test_pred = self.model.predict(self.X_train)
        rmse = mean_squared_error(self.y_train, y_test_pred, squared=False)
        return round(rmse, 2)

    def calculate_r_squared(self):
        y_test_pred = self.model.predict(self.X_train)
        r_squared = r2_score(self.y_train, y_test_pred)
        return round(r_squared, 4)

    def calculate_mae(self):
        y_test_pred = self.model.predict(self.X_train)
        mae = mean_absolute_error(self.y_train, y_test_pred)
        return round(mae, 2)

    def show_data_info(self):
        if self.df is not None:
            numeric_columns = self.df.select_dtypes(include=["number"]).columns
            corr_matrix = self.df[numeric_columns].corr()

            info_text = f"Data Info:\n{self.df.info()}\n\nData Description:\n{self.df.describe()}\n\nCorrelation Matrix:\n{corr_matrix}"

            # Heatmap
            heatmap_fig, heatmap_ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=heatmap_ax
            )
            heatmap_ax.set_title("Correlation Heatmap")

            heatmap_fig.tight_layout()
            heatmap_fig.show()

            self.text_browser.setPlainText(info_text)

    def plot_histogram(self):
        plt.figure(figsize=(8, 6))
        sns.histplot(self.df["Price"], kde=True)
        plt.title("Distribution of Laptop Prices")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.show()

    def plot_pairplot(self):
        features_to_plot = ["Inches", "cpu_speed", "Weight_kg", "Price"]
        sns.pairplot(self.df[features_to_plot])
        plt.suptitle("Pair Plot of Selected Features")
        plt.show()

    def ssd_checkbox_state_changed(self, state):
        # Always enable the storage size combo box
        self.ssd_size_combobox.setDisabled(False)


class RecommendationCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)

    def plot_scatter(self, df, input_data, model):
        self.ax.clear()

        self.ax.scatter(
            df["cpu_speed"].abs(),
            df["Price"],
            label="Training Data",
            color="blue",
        )

        self.ax.scatter(
            input_data["cpu_speed"].abs(),
            model.predict(input_data),
            label="Predicted Price",
            color="red",
        )

        self.ax.set_xlabel("CPU Speed")
        self.ax.set_ylabel("Price")
        self.ax.legend()

        self.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = LaptopPricePredictionApp()
    main_window.show()
    sys.exit(app.exec())