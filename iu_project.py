import sqlalchemy as db
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show 
from bokeh.layouts import column
import streamlit as st

class DatabaseManager:
    """
    A class to manage database operations for training, test, and ideal function data.
    
    Attributes:
    engine (sqlalchemy.engine.Engine): The SQLAlchemy engine for database connection.
    conn (sqlalchemy.engine.Connection): The connection object for executing SQL commands.
    metadata (sqlalchemy.MetaData): The metadata object for handling table schemas.
    """
    def __init__(self, db_url="sqlite:///functions.sqlite"):
        """
        Initialises the DatabaseManager with a connection to the specified SQLite database.
        
        Parameters:
        db_url (str): The URL of the database.
        """
        # Initialise database connection and setup tables
        self.engine = db.create_engine(db_url)
        self.conn = self.engine.connect()
        self.metadata = db.MetaData()
        self.setup_tables()

    def setup_tables(self):
        """
        Sets up the tables for training data, test data, and ideal functions in the database.
        """
        self.training_data = db.Table('training_data', self.metadata,
            db.Column('x', db.Float),
            db.Column('y1', db.Float),
            db.Column('y2', db.Float),
            db.Column('y3', db.Float),
            db.Column('y4', db.Float))
        
        self.test_data = db.Table('test_data', self.metadata,
            db.Column('x', db.Float),
            db.Column('y', db.Float))
        
        columns = [db.Column('x', db.Float)] + [db.Column(f'y{i}', db.Float) for i in range(1, 51)]
        self.ideal_functions = db.Table('ideal_functions', self.metadata, *columns)

        # Create tables in the database
        self.metadata.create_all(self.engine)
        
        # Load table definitions from the database
        self.training_data = db.Table('training_data', self.metadata, autoload_with=self.engine)
        self.ideal_functions = db.Table('ideal_functions', self.metadata, autoload_with=self.engine)
        self.test_data = db.Table('test_data', self.metadata, autoload_with=self.engine)
    
    def load_csv_to_db(self, file_path, table_name):
        """
        Loads CSV data into the specified table in the database.
        
        Parameters:
        file_path (str): The path to the CSV file.
        table_name (str): The name of the table to load data into.
        """
        df = pd.read_csv(file_path)
        df.to_sql(table_name, con=self.engine, if_exists='replace', index=False)
    
    def fetch_data(self, table_name):
        """
        Fetches data from the specified table in the database.
        
        Parameters:
        table_name (str): The name of the table to fetch data from.
        
        Returns:
        pandas.DataFrame: The data from the table as a DataFrame.
        """
        return pd.read_sql_table(table_name, con=self.engine)    

class DataVisualiser(DatabaseManager):
    """
    A class to manage data visualization, inheriting from DatabaseManager.
    
    Attributes:
    best_fit_functions (list): The list of best fit ideal functions.
    max_deviations (list): The list of maximum deviations for each best fit function.
    """
    def __init__(self, db_url="sqlite:///functions.sqlite"):
        """
        Initialises the DataVisualiser with a connection to the specified SQLite database.
        
        Parameters:
        db_url (str): The URL of the database.
        """
        super().__init__(db_url)
    
    def load_data(self, test_data_file):
        """
        Loads training, ideal, and test data into the database.
        
        Parameters:
        test_data_file (str): The path to the test data CSV file.
        """
        self.load_csv_to_db('Dataset2/train.csv', 'training_data')
        self.load_csv_to_db('Dataset2/ideal.csv', 'ideal_functions')
        if test_data_file is not None:
            # Load test data only if a file is provided
            df = pd.read_csv(test_data_file)
            df.to_sql('test_data', con=self.engine, if_exists='replace', index=False)

    def fetch_all_data(self):
        """
        Fetches all necessary datasets (training, test, and ideal functions) from the database.
        """
        self.dataset_training = self.fetch_data('training_data')
        self.dataset_test = self.fetch_data('test_data')
        self.dataset_ideal = self.fetch_data('ideal_functions')
    
    def find_best_fit_functions(self, threshold):
        """
        Finds the best fitting ideal functions for the training data based on a threshold.
        
        Parameters:
        threshold (float): The threshold multiplier for deviation calculation.
        """
        training_df = self.dataset_training
        ideal_df = self.dataset_ideal

        best_fits = []
        max_deviations = []
        # Iterate over each column in the training data (y1 to y4).
        for col in ['y1', 'y2', 'y3', 'y4']:
            errors = []
            deviations = []
            # Iterate over each column in the ideal functions dataset (excluding the first column 'x').
            for ideal_col in ideal_df.columns[1:]:
                # Calculate the sum of squared errors between the current training column and ideal function column.
                error = np.sum((training_df[col] - ideal_df[ideal_col])**2)
                # Calculate the maximum absolute deviation between the current training column and ideal function column.
                max_deviation = np.max(np.abs(training_df[col] - ideal_df[ideal_col]))
                # Append error and max deviation.
                errors.append((ideal_col, error))
                deviations.append((ideal_col, max_deviation))
            # Identify ideal function with minimum error for the current training column.
            best_fit = min(errors, key=lambda x: x[1])[0]
            best_fits.append(best_fit)
            # Identify minimum maximum deviation for the current training column.
            max_deviation = min(deviations, key=lambda x: x[1])[1]
            max_deviations.append(max_deviation * threshold)
        # Store the best fitting functions and their max deviations in the instance variables.
        self.best_fit_functions = best_fits
        self.max_deviations = max_deviations

    def map_test_data_to_functions(self):
        """
        Maps test data to the best fitting ideal functions based on deviation thresholds.
        """
        # Retrieve test and ideal datasets from instance variables.
        test_df = self.dataset_test
        ideal_df = self.dataset_ideal

        results = []
        # Iterate over each row in the test dataset.
        for index, row in test_df.iterrows():
            x_value = row['x']
            y_value = row['y']
            best_fit = None
            min_deviation = None
            # Iterate over each ideal function identified as the best fit.
            for i, ideal_col in enumerate(self.best_fit_functions):
                # Retrieve the ideal y value corresponding to the x value in the test data.
                ideal_y_value = ideal_df[ideal_col].loc[ideal_df['x'] == x_value].values[0]
                deviation = abs(y_value - ideal_y_value)
                # Check if the deviation is within the allowed maximum deviation threshold.
                if deviation <= self.max_deviations[i] * np.sqrt(2):
                    # Update the best fit function if the current deviation is the smallest found.
                    if min_deviation is None or deviation < min_deviation:
                        best_fit = i + 1
                        min_deviation = deviation
            results.append((x_value, y_value, min_deviation, best_fit))

        # Save the results to the database
        results_df = pd.DataFrame(results, columns=['x', 'y', 'delta_y', 'ideal_function'])
        results_df.to_sql('test_results', con=self.engine, if_exists='replace', index=False)

    def visualise_data(self, show_filtered):
        """
        Visualises the training data, ideal functions, and test data using Bokeh and Streamlit.
        
        Parameters:
        show_filtered (bool): Whether to show only filtered test data.
        """
        training_df = self.dataset_training
        ideal_df = self.dataset_ideal
        # Change displayed data base on variable "show_filtered".
        if show_filtered:
            test_results_df = self.fetch_data('test_results')
            test_results_df = test_results_df[test_results_df['ideal_function'].notnull()]
        else:
            test_results_df = self.fetch_data('test_results')

        # Configure plots.
        p1 = figure(title="Training Data vs Ideal Functions", x_axis_label='X', y_axis_label='Y')
        colors = ['red', 'blue', 'green', 'orange']
        for i, col in enumerate(['y1', 'y2', 'y3', 'y4']):
            p1.line(training_df['x'], training_df[col], legend_label=f'Training {col}', color=colors[i])
            p1.line(ideal_df['x'], ideal_df[self.best_fit_functions[i]], legend_label=f'Ideal {self.best_fit_functions[i]}', color=colors[i], line_dash='dashed')

        p2 = figure(title="Test Data Mapping", x_axis_label='X', y_axis_label='Y')
        p2.scatter(test_results_df['x'], test_results_df['y'], legend_label='Test Data', color='black')
        for i, col in enumerate(self.best_fit_functions):
            p2.line(ideal_df['x'], ideal_df[col], legend_label=f'Ideal {col}', color=colors[i], line_dash='dashed')

        # Display the visualisations using Streamlit
        st.header('Selecting the four best-suited ideal functions', divider='blue')
        st.write('the four best ideal functions selected:')
        with st.expander("Training Data vs Ideal Functions"):
            st.write("Selected Ideal Functions:", self.best_fit_functions)
            st.bokeh_chart(p1, use_container_width=True)

        st.header('Plotting test data and comparison with ideal functions', divider='blue')
        st.write('the test data plotted with the best fit ideal functions:') 
        with st.expander("Test Data Mapping"):
            st.bokeh_chart(p2, use_container_width=True) 

    def run(self, test_data_file, threshold, show_filtered):
        """
        Runs the complete process: load data, process it, and visualise the results.
        
        Parameters:
        test_data_file (str): The path to the test data CSV file.
        threshold (float): The threshold multiplier for deviation calculation.
        show_filtered (bool): Whether to show only filtered test data.
        """
        if test_data_file is not None:
            try:
                # Step 1: Load test data into database.
                self.load_data(test_data_file)
                # Step 2: Fetch all required datasets from database.
                self.fetch_all_data()
                # Step 3: Find best fit ideal functions based on training data.
                self.find_best_fit_functions(threshold)
                # Step 4: Map test data to the identified best fit ideal functions.
                self.map_test_data_to_functions()
                # Step 5: Visualise data.
                self.visualise_data(show_filtered)
            except Exception as e:
                # Handle any exceptions that occur during the process.
                st.error(f"An error occurred: {e}")
            finally:
                # Ensure database connection is closed after the process.
                self.conn.close()
        else:
            # Instruct user to upload test data file, if none is provided.
            st.write("Please upload a test data CSV file.")

def main():
    """
    Main function to run the Streamlit app. Sets up the app interface and handles user input.
    """
    # Set the title and header using Streamlit commands.
    st.title("DLMDSPWP01 – Programming with Python")
    st.header('Technical Assignment', divider='blue')
    # Provide brief overview of the assignment.
    st.write('The objective of this assignment is to develop a Python program to process and analyse a set of training and test data to identify and map ideal functions.') 
    # Detailed description of key tasks and the role of Streamlit.
    st.write('''
        The training data will be used to select the four best-fitting ideal functions from a provided set of fifty.

        The program will then use the test data to map each data point to one of the selected ideal functions based on defined criteria. 

        Key tasks:
        
            - Data Loading and Storage- Using SQLite and SQLAlchemy.
            - Selecting Ideal Functions - Analyze the training data to determine the four ideal functions that best minimise the sum of squared deviations (least-Square method).
            - Mapping Test Data - For each test data point, determine if it can be assigned to one of the four chosen ideal functions.
            - Data Visualisation - All data should be visualised as appropriate using a tool such as Bokeh.
            - Object-Oriented Design - The program should utilise OOP techniques including one example of inheritance.
            - Unit Testing - The program should include unit testing examples.

        Additionally, with the help of Streamlit, I have packaged everything together in this web app, including the features listed below:
             
            - To load the analysis, users must first upload the test data csv file.
            - User can toggle to see all data or filtered data, meaning that only points that have been assigned to one of the chosen ideal functions will be shown.
            - When the above filter is used, users can also select a "Threshold Multiplier" which ajust how sensitive the calculation is, resulting in more or less points being shown in the plot.

    ''')

    # Create Streamlit sidebar configuration panel.
    with st.sidebar:
        st.sidebar.title("Configuration Panel")
        st.sidebar.write("Use the options below to configure the data processing and visualisation settings.")
        # Add file uploader widget using Streamlit.
        st.sidebar.subheader("Upload Test Data")
        test_data_file = st.file_uploader("Upload Test Data CSV", type="csv")
        # Add slider widget using Streamlit.
        st.sidebar.subheader("Adjust Threshold Multiplier")
        st.sidebar.write("The threshold multiplier adjusts how sensitive the mapping is. Higher values make the mapping less strict.")
        threshold = st.slider("Threshold Multiplier", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        # Add checkbox widget using Streamlit.
        st.sidebar.subheader("Toggle Data Display")
        st.sidebar.write("Use this option to switch between displaying all test data and only filtered data points.")
        show_filtered = st.checkbox("Show Filtered Data", value=False)

    # Instantiate the DataVisualiser class.
    visualiser = DataVisualiser()
    # Run the visualiser with provided configuration options.
    visualiser.run(test_data_file, threshold, show_filtered)

# Entry point of the script.
if __name__ == '__main__':
    main()