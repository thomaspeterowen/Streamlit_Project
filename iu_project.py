import sqlalchemy as db
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show 
from bokeh.layouts import column
import streamlit as st

class DatabaseManager:
    def __init__(self, db_url="sqlite:///functions.sqlite"):
        self.engine = db.create_engine(db_url)
        self.conn = self.engine.connect()
        self.metadata = db.MetaData()
        self.setup_tables()

    def setup_tables(self):
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

        self.metadata.create_all(self.engine)
        
        self.training_data = db.Table('training_data', self.metadata, autoload_with=self.engine)
        self.ideal_functions = db.Table('ideal_functions', self.metadata, autoload_with=self.engine)
        self.test_data = db.Table('test_data', self.metadata, autoload_with=self.engine)
    
    def load_csv_to_db(self, file_path, table_name):
        df = pd.read_csv(file_path)
        df.to_sql(table_name, con=self.engine, if_exists='replace', index=False)
    
    def fetch_data(self, table_name):
        return pd.read_sql_table(table_name, con=self.engine)    


class DataVisualizer(DatabaseManager):
    def __init__(self, db_url="sqlite:///functions.sqlite"):
        super().__init__(db_url)
    
    def load_data(self, test_data_file):
        self.load_csv_to_db('Dataset2/train.csv', 'training_data')
        self.load_csv_to_db('Dataset2/ideal.csv', 'ideal_functions')
        #self.load_csv_to_db('Dataset2/test.csv', 'test_data')
        if test_data_file is not None:
            df = pd.read_csv(test_data_file)
            df.to_sql('test_data', con=self.engine, if_exists='replace', index=False)

    def fetch_all_data(self):
        self.dataset_training = self.fetch_data('training_data')
        self.dataset_test = self.fetch_data('test_data')
        self.dataset_ideal = self.fetch_data('ideal_functions')
    
    def find_best_fit_functions(self, threshold):
        training_df = self.dataset_training
        ideal_df = self.dataset_ideal

        best_fits = []
        max_deviations = []
        for col in ['y1', 'y2', 'y3', 'y4']:
            errors = []
            deviations = []
            for ideal_col in ideal_df.columns[1:]:
                error = np.sum((training_df[col] - ideal_df[ideal_col])**2)
                max_deviation = np.max(np.abs(training_df[col] - ideal_df[ideal_col]))
                errors.append((ideal_col, error))
                deviations.append((ideal_col, max_deviation))
            best_fit = min(errors, key=lambda x: x[1])[0]
            best_fits.append(best_fit)
            max_deviation = min(deviations, key=lambda x: x[1])[1]
            max_deviations.append(max_deviation * threshold)
        self.best_fit_functions = best_fits
        self.max_deviations = max_deviations

    def map_test_data_to_functions(self):
        test_df = self.dataset_test
        ideal_df = self.dataset_ideal

        results = []
        for index, row in test_df.iterrows():
            x_value = row['x']
            y_value = row['y']
            best_fit = None
            min_deviation = None
            for i, ideal_col in enumerate(self.best_fit_functions):
                ideal_y_value = ideal_df[ideal_col].loc[ideal_df['x'] == x_value].values[0]
                deviation = abs(y_value - ideal_y_value)
                if deviation <= self.max_deviations[i] * np.sqrt(2):
                    if min_deviation is None or deviation < min_deviation:
                        best_fit = i + 1
                        min_deviation = deviation
            results.append((x_value, y_value, min_deviation, best_fit))

        results_df = pd.DataFrame(results, columns=['x', 'y', 'delta_y', 'ideal_function'])
        results_df.to_sql('test_results', con=self.engine, if_exists='replace', index=False)

    def visualize_data(self, show_filtered):
        training_df = self.dataset_training
        ideal_df = self.dataset_ideal
        if show_filtered:
            test_results_df = self.fetch_data('test_results')
            test_results_df = test_results_df[test_results_df['ideal_function'].notnull()]
        else:
            test_results_df = self.fetch_data('test_results')

        p1 = figure(title="Training Data vs Ideal Functions", x_axis_label='X', y_axis_label='Y')
        colors = ['red', 'blue', 'green', 'orange']
        for i, col in enumerate(['y1', 'y2', 'y3', 'y4']):
            p1.line(training_df['x'], training_df[col], legend_label=f'Training {col}', color=colors[i])
            p1.line(ideal_df['x'], ideal_df[self.best_fit_functions[i]], legend_label=f'Ideal {self.best_fit_functions[i]}', color=colors[i], line_dash='dashed')

        p2 = figure(title="Test Data Mapping", x_axis_label='X', y_axis_label='Y')
        p2.scatter(test_results_df['x'], test_results_df['y'], legend_label='Test Data', color='black')
        for i, col in enumerate(self.best_fit_functions):
            p2.line(ideal_df['x'], ideal_df[col], legend_label=f'Ideal {col}', color=colors[i], line_dash='dashed')

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
        if test_data_file is not None:
            try:
                self.load_data(test_data_file)
                self.fetch_all_data()
                self.find_best_fit_functions(threshold)
                self.map_test_data_to_functions()
                self.visualize_data(show_filtered)
            except Exception as e:
                pass
                st.error(f"An error occurred: {e}")
            finally:
                self.conn.close()
        else:
            st.write("Please upload a test data CSV file.")

def main():
    st.title("DLMDSPWP01 – Programming with Python")
    st.header('Technical Assignment', divider='blue')
    st.write('The objective of this assignment is to develop a Python program to process and analyse a set of training and test data to identify and map ideal functions.') 

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
            - When the above filter is used, users can also select a "Threshold Multiplier" which asjust how sensitive the calculation is, resulting in more or less points being shown in the plot.

    ''')


    with st.sidebar:
        st.sidebar.title("Configuration Panel")
        st.sidebar.write("Use the options below to configure the data processing and visualization settings.")
        
        st.sidebar.subheader("Upload Test Data")
        test_data_file = st.file_uploader("Upload Test Data CSV", type="csv")
        
        st.sidebar.subheader("Adjust Threshold Multiplier")
        st.sidebar.write("The threshold multiplier adjusts how sensitive the mapping is. Higher values make the mapping less strict.")
        threshold = st.slider("Threshold Multiplier", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        
        st.sidebar.subheader("Toggle Data Display")
        st.sidebar.write("Use this option to switch between displaying all test data and only filtered data points.")
        show_filtered = st.checkbox("Show Filtered Data", value=False)

    visualizer = DataVisualizer()
    visualizer.run(test_data_file, threshold, show_filtered)

if __name__ == '__main__':
    main()
