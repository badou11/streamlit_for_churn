import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler

@st.cache
def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL)
    return data

def main_content():
    st.markdown("""
        <h1 style="font-size: 50px; color:#DE781F" >Probleme understanding</h1>
        """, unsafe_allow_html=True)
    st.markdown("""
            Le churn ou attrition est un phénomène qui désigne la résiliation d’un forfait ou d’un contrat qui lie un client à une entreprise. A ce titre, le churn est un indicateur clé pour calculer la satisfaction de la clientèle.
            Dans cette exercice, on dispose des données d'une banque virtuelle qui a remarquée que beaucoup de ses clients quittés la banque.
            L'ojectif est donc de comprendre le phenomeme qui améne ses clients à quitter la banque et pour cela, on dispose d'un dataset contenant une colonne nous informant si le client à quitter la banque ou non.

            ## Description du jeu de donnée

            - `RowNumber`: le numero de la ligne de donnée
            - `CustomerId`: identifiant du client
            - `Surname`: Surnom du client
            - `CreditScore`: Le score de credit nous donne une idée de la capacité de remboursement d'un client
            - `Geography`: nom du pays de provenance du client
            - `Gender`: Le genre du client
            - `Age`: L'age du client
            - `Tenure`: Nombre d'année depuis laquelle la personne est presente dans la banque
            - `Balance`: Quantité d'argent qui est sur le compte du client
            - `NumOfProducts`: Nombre de produit que le client a dans la banque
            - `HasCrCard`: Si le client a une carte de credit ou non
            - `IsActiveMember`: Designe si le client est un membre actif de la banque
            - `EstimatedSalary`: Le salaire estimé de chaque client 
            - `Exited`: la colonne Target qu'il nous faut predire la valeur

            ## Methode de travail

            Pour regler le probleme, on va suivre une methode de travail qui va nous permettre d'atteindre les objectifs.

            ### 1. Analyse Exploratoire des données
            ### 2. Pré-traitement des données
            ### 3. Modelisation
            ### 4. Evaluation du modele
            ### 5. Mise en production avec une API flask
        """)


def file_selector(folder_path='./datasets'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("select a file", filenames)

    return os.path.join(folder_path, selected_filename)


# show the data
def show_raw_data(data):
    number = st.number_input("Number of row to view", 5, 100, 5)
    "Raw Data", data.head(number)

def columns_info_and_distribution(data):
    selected_columns = st.multiselect("Select", list(data.columns), default=['RowNumber'])
    data[selected_columns]
    if 'Exited' not in selected_columns:
        st.subheader("distribution curve")
        for col in selected_columns:
            if str(data[col].dtypes) == 'object':
                st.text("Can't display the distribution plot of a categorical variable") 
            else:
                fig, ax = plt.subplots()
                fig = plt.figure(figsize=(12, 8))
                ax = plt.axvline(x=data[col].quantile(q=0.25), c='C1',linestyle=':')
                ax = plt.axvline(x=data[col].quantile(q=0.75), c='C1',linestyle=':')
                ax = plt.axvline(x=data[col].mean(), c='C1')
                ax = plt.axvline(x=data[col].median(), c='C1',linestyle='--')

                ax = plt.hist(data[col], bins=100, histtype='step', density=True)
                ax = data[col].plot.density(bw_method=0.5)
                plt.legend()
                st.pyplot(fig)
    else:
        st.subheader("distribution curve between target and variable")
        for col in selected_columns:
            if str(data[col].dtypes) == 'object':
                st.text("Can't display the distribution plot of a categorical variable") 
            else:
                fig, ax = plt.subplots()
                fig = plt.figure(figsize=(16, 9))
                ax = sns.distplot(data[data['Exited']==1][col], label="Exited")
                ax = sns.distplot(data[data['Exited']==0][col], label="Stayed")
                plt.legend()
                st.pyplot(fig)

def target_info(data):
    st.text('Value Counts By Target/Class')
    st.write(data['Exited'].value_counts(normalize=True))
    st.subheader("Pie chart")
    fig, ax = plt.subplots()
    st.write(data.iloc[:, -1].value_counts().plot.pie())
    ax = data.iloc[:, -1].value_counts().plot.pie(autopct='%.1f%%', labels = ['Stayed', 'Exited'], figsize =(5,5), 
                                                fontsize = 10)
    plt.title('Statistique de Churn', fontsize = 11)

    st.pyplot(fig)


def customized_plot(data):
    st.subheader("Data Visualization")
    type_of_plot = st.selectbox("Select type of plot", ["bar", "area", "line", "hist", "boxplot", "kde"])
    selected_columns_names = st.multiselect("Select a colomn", data.columns.tolist())
    if st.button('Generate Plot'):
        st.success(f"Generating {type_of_plot} plot for {selected_columns_names}")

    if type_of_plot == "area":
        cust_data = data[selected_columns_names]
        st.area_chart(cust_data)

    elif type_of_plot == "bar":
        cust_data = data[selected_columns_names]
        st.bar_chart(cust_data)

    elif type_of_plot == "line":
        cust_data = data[selected_columns_names]
        st.line_chart(cust_data)

    elif type_of_plot:
        fig, ax = plt.subplots(figsize=(20, 20))
        ax = data[selected_columns_names].plot(kind=type_of_plot)
        st.pyplot(fig)

def corr_matrix(data):
    fig, ax = plt.subplots()
    ax = sns.heatmap(data.corr(), annot=True, cbar=False)
    st.pyplot(fig)

def countplots(data):
    columns = ['Age', 'Geography', 'Gender', 'IsActiveMember']
    choice = st.selectbox("Select a columns", columns)
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(16, 9))
    ax = sns.countplot(x=choice, data=data, hue='Exited')
    st.pyplot(fig)
    
    if choice != 'Age':
        fig2, ax2 = plt.subplots()
        ax2 = sns.heatmap(pd.crosstab(data.Exited, data[choice], normalize='columns'), annot=True)
        st.pyplot(fig2)

def eda_content():
    st.title("1. Exploratory Data Analysis")
    st.subheader("In this part of our App, we'll explore our data with streamlit.")

    filename = file_selector()
    st.info("loading ... "+filename)
    data = load_data(filename)
    st.success('Your data has been loaded succesfully :smile:')
  
    if st.checkbox("Show Dataset"):
        show_raw_data(data)

    # Show columns
    if st.checkbox("Show Columns details"):
        "Dataset Details"
        st.write(data.dtypes)

    # Show data shape
    if st.checkbox("Show our Shape"):
        data.shape


    # Select columns
    if st.checkbox("Select some columns"):
        columns_info_and_distribution(data)

    # show values
    if st.button('Value Counts for the Target Variable'):
        # target_info is a function that show some info about the target variable.
        target_info(data)

    # Show some Stats
    if st.button('Some Stats !!'):
        st.write(data.describe())

    # Visualization
    
    # customized plot
    customized_plot(data)
        
    # Crrelation Map
    if st.checkbox('Correlation Map'):
        corr_matrix(data)
        

    # Count plot
    if st.checkbox('Count plot'):
        countplots(data)
    


def activity():
    activity = ['Main', 'Exploratory Data Analysis', 'Modelisation', 'Deploy the API']
    choice = st.sidebar.selectbox("Choose Activity", activity)
    return choice


def create_dummy_and_encoding(data):
    col_to_drop = ['RowNumber', 'CustomerId', 'Surname'] 
    for col in col_to_drop:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)
    data.loc[: ,'Gender'] = data.loc[:, 'Gender'].map({'Female':0, 'Male':1})
    data = pd.get_dummies(data=data, drop_first=True)
    return data

def imputation(df):
    return df.dropna(axis=0)


def preprocessing(data): 
    data = create_dummy_and_encoding(data)
    data = imputation(data)
    X = data.drop('Exited', axis=1)
    y = data.Exited
    
    print(y.value_counts())
    
    return X, y

def load_model():
    with open('model/RandomForest', 'rb') as f:
        RandomForest = pickle.load(f)
    with open('model/XGBoostClassifier', 'rb') as f:
        XGBOOST = pickle.load(f)
    
    return RandomForest, XGBOOST


def evaluation(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    ypred =model.predict(X_test)

    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
    
    N, train_scores, test_scores = learning_curve(model, X_train, y_train, train_sizes = np.linspace(0.1, 1, 10), 
                                                  cv=10, scoring='f1')
    
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_scores.mean(axis=1), label='train score')
    plt.plot(N, test_scores.mean(axis=1), label='test score')
    plt.legend()
    plt.show()


def modelisation_content(data):
    st.markdown("""
        <h1 style="font-size: 50px; color:#DE781F" >Modelisation</h1>
        <div>We find in our previous analysis(EDA) that wa have an unbalanced Dataset. So, we can perform SMOTE Analysis to solve the issue or use another metric like <em style = "font-weight:bold; color:#1F85DE">Precision</em>, <em style = "color:#1F85DE; font-weight :bold">Recall</em> or <em style = "color:#1F85DE; font-weight :bold">F1_Score</em> if we want to be more rigourous.</div>
        """, unsafe_allow_html=True)
    st.sidebar.subheader("Train set size")
    train_size = st.sidebar.slider("Train", min_value=0.2, max_value=0.9, value=0.8)
    
    # if st.sidebar.button("Submit"):
    smote = st.sidebar.radio("Smote Analysis ?", ["Yes", "No"])
    list_of_algo = st.sidebar.multiselect("choose an algorithm", ['Random Forest Classifier', 'AdaBoostClassifier', 'XGBOOSTClassifer','KNN', 'SVM'])

    st.sidebar.text('Random Forest and XGBOOST are pre-trained and they show the best perofrmance for this particular problem !!')
    test_size = 1 - float(train_size)
    trainset, testset = train_test_split(data, test_size=test_size, random_state=0)
    
    X_train, y_train = preprocessing(trainset)
    X_test, y_test = preprocessing(testset)

    st.subheader("Trainset after Preprocessing")
    st.write(X_train.head(5))
    
    st.subheader("Target Variable")
    st.write(y_train.head(5))

    if ('Random Forest Classifier' or 'XGBOOSTClassifer') in list_of_algo:
        predict = st.sidebar.button('PREDICT')
        if predict:
            RandomForest, XGBOOST = load_model()
            y_pred_random_forest = RandomForest.predict(X_test)
            y_pred_xgboost = XGBOOST.predict(X_test)
            st.subheader("Predicted values with RandomForest")
            val = pd.DataFrame({'True value':y_test, 'Predicted Values':y_pred_random_forest})
            st.dataframe(val.style.highlight_max(axis=0))
            st.help(st.dataframe)
            st.subheader("Predicted values with XGBOOST")
            st.write(y_pred_xgboost)





def main():
    """Common Machine Learning EDA"""
    choice = activity()
    if choice == 'Main':
        main_content()
    elif choice == "Exploratory Data Analysis":
        eda_content()
    elif choice == "Modelisation":
        filename = file_selector()
        data = load_data(filename)
        modelisation_content(data)


if __name__ == '__main__':
    main()

# def highlight_max(df):
#     '''
#     highlight the maximum in a Series yellow.
#     '''
#     if df.iloc[:, 0] == df.iloc[:, 1]
        
#         return ['background-color: yellow' if v else '' for v in is_max]