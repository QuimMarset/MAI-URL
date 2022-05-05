import pandas as pd
from sklearn import preprocessing
from scipy.io import arff


def read_dataset(dataset_path):
    data, _ = arff.loadarff(dataset_path)
    df = pd.DataFrame(data)
    df = df.applymap(lambda cell: cell.decode('utf-8') if isinstance(cell, bytes) else cell)
    return df


def normalize_numeric_features(numeric_features):
    standard_scaler = preprocessing.StandardSMcaler()
    scaled_numeric_features = standard_scaler.fit_transform(numeric_features)
    scaled_numeric_features = pd.DataFrame(scaled_numeric_features, columns=numeric_features.columns)
    return scaled_numeric_features


def convert_categorical_to_one_hot(categorical_features):
    one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)
    one_hot_features = one_hot_encoder.fit_transform(categorical_features)
    columns = one_hot_encoder.get_feature_names(categorical_features.columns)
    encoded_categorical_features = pd.DataFrame(one_hot_features, columns=columns)
    return encoded_categorical_features


def handle_numeric_missings(df):
    # In this dataset we only observe missings in 'capital-gain' with the value 999999
    capital_gain_feature = df['capital-gain'].copy()
    mean_value = capital_gain_feature.loc[capital_gain_feature < 90000].mean()
    capital_gain_feature.loc[capital_gain_feature > 90000] = mean_value
    df['capital-gain'] = capital_gain_feature
    return df


def handle_categorical_missings(df):
    #There are missings in 'occupation', 'workclass', 'native-country' represented as '?'
    modes = df[['occupation', 'workclass', 'native-country']].mode()
    df['occupation'].replace({'?': modes['occupation'][0]}, inplace=True)
    df['workclass'].replace({'?': modes['workclass'][0]}, inplace=True)
    df['native-country'].replace({'?': modes['native-country'][0]}, inplace=True)
    return df


def preprocess_adult(arff_path):
    # Reads the dataset from the arff file, converting it to a pandas Dataframe
    df = read_dataset(arff_path)

    # Extract the class column from the DataFrame
    df_class_feature = df['class'].copy()
    # Remove it from the dataset
    df = df.drop('class', axis=1)
    
    # Convert the label into a number
    df_class_feature_numeric = df_class_feature.replace({'>50K' : 1, '<=50K' : 0})

    # Impute the missing values in the needed features
    df = handle_numeric_missings(df)
    df = handle_categorical_missings(df)
    
    # Separate the numeric and categorical features
    numeric_features = df.select_dtypes(include=['number'])
    categorical_features = df.select_dtypes(include=[object])

    # Normalize numeric features to the range [0, 1]
    scaled_numeric_features = normalize_numeric_features(numeric_features)

    # Convert cardinal categorical features to numeric using One Hot Encoding. 'education' is repeated with 'education-num'
    columns_to_one_hot = [column for column in categorical_features.columns if column not in ['education', 'sex']]
    one_hot_features = convert_categorical_to_one_hot(categorical_features[columns_to_one_hot])
    
    # This feature is not ordinal, but we can use Label Encoding it as it only has 2 categories
    encoded_sex_feature = categorical_features['sex'].replace({'Male' : 0, 'Female' : 1})

    df_all_numeric = pd.concat([scaled_numeric_features, one_hot_features, encoded_sex_feature], axis=1)
    return df_all_numeric.to_numpy(), df_class_feature_numeric.to_numpy()