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

@st.cache
def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

def main():
    """Common Machine Learning EDA"""
    
    st.sidebar.markdown("""
    <h2 style="font-size: 30px; color:#1F85DE" >Objectif : </h2>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""

        - `Objectif` : ``Comprendre notre dataset``

        **Visualisation de la target** :
        - Notre dataset n'est pas equilibré 
        - 79,6% n'ont pas churner
        - 20,4% ont churner
        - Il faut utiliser des metriques comme le `F1-Score`, la `precision` ou le `Recall`ou l'analyse `smote` pour regler le probleme des classes desiquilibrées
    
        **Signification des variables** :
         - **Variable categorique**
        - France : 51,1%
        - Germany : 25,1%
        - Spain : 24,8%
 
        - **Relation Variables / Target** :        
            - Nos données n'ont pas été normalisée, mais ils ont l'aire de suivre une loi normale.
            - Il semblerait que l'age joue sur le churn : ``(hypothese)``
            - Les hommes sont plus representé que les femmes et la banque a beaucoup de client en France/
            - Les clients qui sont en Allemagne churn le plus. 
            - Les femmes churn plus que les hommes. 
        
        ## Analyse plus détaillée

        - **Relation Variables / Variables** :
            - On voit que certain variable sont tres fortement correlé
        
        ### hypotheses nulle (H0): 
        - **Resultat de l'analyse: **
            - Les individues qui churn le plus sont agés entre 40 et 60 ans
            - H0 : le taux de churn moyen est egale dans tout les tranches d'ages
                - Cette hypothese est rejetés. Donc l'age a un effet important sur le churn et ceux qui churn le plus son entre 40 et 60 ans 

    """)


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
    st.title("1. Exploratory Data Analysis")
    st.subheader("In this part of our App, we'll explore our data with streamlit.")

    # EDA_button = st.button('Let\'s begin ...')

    def file_selector(folder_path='./datasets'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox("select a file", filenames)

        return os.path.join(folder_path, selected_filename)

    filename = file_selector()
    st.info("loading ... "+filename)
    data = load_data(filename)
    st.success('Your data has been loaded succesfully :smile:')
    # st.balloons()
    # time.sleep(1)
    # st.text("Let's continue ...")

    # st.help(st.multiselect)
    # show the data
    if st.checkbox("Show Dataset"):
        # st.text("Hello world")
        number = st.number_input("Number of row to view", 5, 100, 5)
        "Raw Data", data.head(number)
    
    # Show columns
    if st.checkbox("Show Columns details"):
        "Dataset Details"
        st.write(data.dtypes)
    
    # Show data shape
    if st.checkbox("Show our Shape"):
        "Dataset shape"
        data.shape

    # Select columns
    if st.checkbox("Select some columns"):
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


    # show values
    if st.button('Value Counts for the Target Variable'):
        st.text('Value Counts By Target/Class')
        st.write(data['Exited'].value_counts(normalize=True))
        st.subheader("Pie chart")
        fig, ax = plt.subplots()
        st.write(data.iloc[:, -1].value_counts().plot.pie())
        ax = data.iloc[:, -1].value_counts().plot.pie(autopct='%.1f%%', labels = ['Stayed', 'Exited'], figsize =(5,5), 
                                                  fontsize = 10)
        plt.title('Statistique de Churn', fontsize = 11)
  
        st.pyplot(fig)

    # Show some Stats
    if st.button('Some Stats !!'):
        st.write(data.describe())

    # Visualization

    # customized plot
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

    if st.checkbox('Correlation Map'):
        fig, ax = plt.subplots()
        ax = sns.heatmap(data.corr(), annot=True, cbar=False)
        st.pyplot(fig)

    # Count plot
    if st.checkbox('Count plot'):
        columns = ['Age', 'Geography', 'Gender', 'IsActiveMember']
        choice = st.selectbox("Select a columns", columns)
        fig, ax = plt.subplots()
        fig = plt.figure(figsize=(16, 9))
        ax = sns.countplot(x=choice, data=data, hue='Exited')
        
        # fig = px.histogram(data, x=data[choice])
        # st.plotly_chart(fig)

        st.pyplot(fig)
        
        if choice != 'Age':
            fig2, ax2 = plt.subplots()
            ax2 = sns.heatmap(pd.crosstab(data.Exited, data[choice], normalize='columns'), annot=True)
            st.pyplot(fig2)
        

        



if __name__ == '__main__':
    main()