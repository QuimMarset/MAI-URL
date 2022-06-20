import pandas as pd
from sklearn import preprocessing
from scipy.io import arff
from utils.plot_utils import write_cluster_centers_to_txt
import numpy as np
from utils.string_utils import ADULT


class AdultData:

    def __init__(self, dataset_path, save_path):
        self.save_path = save_path
        self.encoder = preprocessing.OneHotEncoder(sparse=False)
        self.scaler = preprocessing.StandardScaler()
        self.preprocess_dataset(dataset_path)


    def read_dataset(self, dataset_path):
        data, _ = arff.loadarff(dataset_path)
        df = pd.DataFrame(data)
        df = df.applymap(lambda cell: cell.decode('utf-8') if isinstance(cell, bytes) else cell)
        return df


    def get_original_cluster_center(self, df_cluster):
        num_statistics = df_cluster.describe()
        means = num_statistics.loc['mean'].to_numpy()
        modes = df_cluster.select_dtypes(include=[object]).mode()
        feature_names_numeric = num_statistics.columns.to_numpy()
        feature_names_categoric = modes.columns.to_numpy()
        return np.concatenate((means, modes.to_numpy()[0])), np.concatenate((feature_names_numeric, feature_names_categoric))


    def save_original_cluster_centers(self, df):
        cluster_1 = df.loc[df['class'] == '<=50K']
        cluster_2 = df.loc[df['class'] == '>50K']
        center_1, feature_names = self.get_original_cluster_center(cluster_1)
        center_2, _ = self.get_original_cluster_center(cluster_2)
        centers = np.array([center_1, center_2])
        write_cluster_centers_to_txt(centers, feature_names, 'Ground-truth', ADULT, self.save_path)


    def normalize_numeric_features(self, numeric_features):
        scaled_numeric_features = self.scaler.fit_transform(numeric_features)
        scaled_numeric_features = pd.DataFrame(scaled_numeric_features, columns=numeric_features.columns)
        return scaled_numeric_features


    def convert_categorical_to_one_hot(self, categorical_features):
        one_hot_features = self.encoder.fit_transform(categorical_features)
        columns = self.encoder.get_feature_names_out(categorical_features.columns)
        encoded_categorical_features = pd.DataFrame(one_hot_features, columns=columns)
        return encoded_categorical_features


    def handle_numeric_missings(self, df):
        # In this dataset we only observe missings in 'capital-gain' with the value 999999
        capital_gain_feature = df['capital-gain'].copy()
        mean_value = capital_gain_feature.loc[capital_gain_feature < 90000].mean()
        capital_gain_feature.loc[capital_gain_feature > 90000] = mean_value
        df['capital-gain'] = capital_gain_feature
        return df


    def handle_categorical_missings(self, df):
        #There are missings in 'occupation', 'workclass', 'native-country' represented as '?'
        modes = df[['occupation', 'workclass', 'native-country']].mode()
        df['occupation'].replace({'?': modes['occupation'][0]}, inplace=True)
        df['workclass'].replace({'?': modes['workclass'][0]}, inplace=True)
        df['native-country'].replace({'?': modes['native-country'][0]}, inplace=True)
        return df


    def preprocess_dataset(self, arff_path):
        # Reads the dataset from the arff file, converting it to a pandas Dataframe
        df = self.read_dataset(arff_path)

        # Extract the class column from the DataFrame
        df_class_feature = df['class'].copy()
        # Convert the label into a number
        df_class_feature_numeric = df_class_feature.replace({'>50K' : 1, '<=50K' : 0})
        self.true_clustering = df_class_feature_numeric.to_numpy()

        # Remove 'fnlwgt' (does not give any valuable information) and 'education' (is repeated with 'education-num' when transformed to numeric)
        df = df.drop(['fnlwgt', 'education'], axis=1)

        # Impute the missing values in the needed features
        df = self.handle_numeric_missings(df)
        df = self.handle_categorical_missings(df)

        # Get the original cluster centers
        self.save_original_cluster_centers(df)

        # Remove the target class from the dataset
        df = df.drop('class', axis=1)
        
        # Separate the numeric and categorical features
        numeric_features = df.select_dtypes(include=['number'])
        categorical_features = df.select_dtypes(include=[object])

        numeric_feature_names = numeric_features.columns.to_numpy()
        categ_feature_names = categorical_features.columns.to_numpy()
        self.column_names = np.concatenate((numeric_feature_names, categ_feature_names[categ_feature_names != 'sex'], ['sex']))

        # Normalize numeric features to the range [0, 1]
        scaled_numeric_features = self.normalize_numeric_features(numeric_features)

        # Convert cardinal categorical features to numeric using One Hot Encoding
        columns_to_one_hot = categ_feature_names[categ_feature_names != 'sex']
        one_hot_features = self.convert_categorical_to_one_hot(categorical_features[columns_to_one_hot])
        
        # This feature is not ordinal, but we can use Label Encoding it as it only has 2 categories
        encoded_sex_feature = categorical_features['sex'].replace({'Male' : 0, 'Female' : 1})

        self.data_with_target = pd.concat([scaled_numeric_features, one_hot_features, encoded_sex_feature, df_class_feature], axis=1)
        self.data = pd.concat([scaled_numeric_features, one_hot_features, encoded_sex_feature], axis=1)


    def get_content(self):
        return self.data.to_numpy(), self.true_clustering, self.column_names


    def restore_center_features(self, centers):
        num_centers = centers.shape[0]
        
        # The centers come after restoring the features before applying PCA
        # However, we need to get them right to restore them from applying one-hot encoding and normalization

        centers_categorical = centers[:, 5:-1]

        # Select the index corresponding to the category that will have a 1 in the one-hot feature
        workclass_values = np.argmax(centers_categorical[:, :8], axis=1)
        marital_status_values = np.argmax(centers_categorical[:, 8:15], axis=1)
        occupation_values = np.argmax(centers_categorical[:, 15:29], axis=1)
        relationship_values = np.argmax(centers_categorical[:, 29:35], axis=1)
        race_values = np.argmax(centers_categorical[:, 35:40], axis=1)
        native_country_values = np.argmax(centers_categorical[:, 40:], axis=1)

        indices = np.concatenate((workclass_values[:, np.newaxis], marital_status_values[:, np.newaxis]+8, 
            occupation_values[:, np.newaxis]+15, relationship_values[:, np.newaxis]+29, 
            race_values[:, np.newaxis]+35, native_country_values[:, np.newaxis]+40), axis=1)

        centers_one_hot = np.zeros_like(centers_categorical)
        # Set to one the corresponding features
        centers_one_hot[np.arange(indices.shape[0])[:, np.newaxis], indices] = 1

        # Transform back the categorical features to the original space
        centers_categoric = self.encoder.inverse_transform(centers_one_hot)
        # Transform back the numeric features previous to normalizing
        centers_numeric = self.scaler.inverse_transform(centers[:, :5])

        restored_centers = np.zeros((num_centers, centers_numeric.shape[1]+centers_categoric.shape[1]+1), dtype=object)
        restored_centers[:, :5] = centers_numeric
        restored_centers[:, 5:-1] = centers_categoric
        # Add 'sex' feature as is. It only has 2 categories and it is easy to interpret
        restored_centers[:, -1] = centers[:, -1]
        return restored_centers