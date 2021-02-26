import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import base64
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import plotly.figure_factory as ff
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from sklearn.model_selection import train_test_split
import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# --------------------------------------------


@st.cache
def load_data(uploaded):
    return pd.read_csv(uploaded)


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a style="font-size: 10px; color: purple; text-decoration: none;" href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def customized_plot(type_of_plot, columns, data, target, bins=0):

    if type_of_plot == "Scatter":
        if len(columns) > 1 and len(columns) <= 2:
            fig = px.scatter(
                data, x=columns[0], y=columns[1], width=620, height=420, title="Evolution of "+columns[0]+" according to " + columns[1])

            fig.update_layout(title_x=0.5, font_size=15)
            st.plotly_chart(fig)
        else:
            st.sidebar.error('Choose until 2 columns')

    if type_of_plot == "Bar":
        if len(columns) > 1 and len(columns) <= 2:
            fig = px.bar(data_frame=data, x=columns[0], y=columns[1],
                         width=620, height=420, barmode="relative")
            st.plotly_chart(fig)
        else:
            st.sidebar.error('Choose until 2 columns')

    if type_of_plot == "Countplot":
        fig, ax = plt.subplots()
        fig = plt.figure(figsize=(16, 9))
        ax = sns.countplot(x=columns, data=data, hue=target)
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2 = sns.heatmap(pd.crosstab(
            data[target], data[columns], normalize='columns'), annot=True)
        st.pyplot(fig2)

    if type_of_plot == "Boxplot":
        if len(columns) > 1:
            fig = px.box(data_frame=data, x=columns[0], y=columns[1])
        else:
            fig = px.box(data_frame=data, y=columns,
                         width=620, height=420,  orientation="v")

        st.plotly_chart(fig)

    if type_of_plot == "Histogram":
        fig = px.histogram(data_frame=data, x=columns,
                           nbins=int(bins), width=620, height=420, title="Distribution of "+columns)
        fig.update_layout(title_x=0.5, font_size=15)
        st.plotly_chart(fig)

    if type_of_plot == "Distribution":
        if target not in columns:
            st.subheader("distribution curve")
            for col in columns:
                if str(data[col].dtypes) == 'object':
                    st.text(
                        "Can't display the distribution plot of a categorical variable")
                else:
                    fig, ax = plt.subplots()
                    fig = plt.figure(figsize=(12, 8))
                    ax = plt.axvline(x=data[col].quantile(
                        q=0.25), c='C1', linestyle=':')
                    ax = plt.axvline(x=data[col].quantile(
                        q=0.75), c='C1', linestyle=':')
                    ax = plt.axvline(x=data[col].mean(), c='C1')
                    ax = plt.axvline(
                        x=data[col].median(), c='C1', linestyle='--')

                    ax = plt.hist(data[col], bins=100,
                                  histtype='step', density=True)
                    ax = data[col].plot.density(bw_method=0.5)

                    plt.legend()
                    st.pyplot(fig)
        else:
            st.subheader("distribution curve between target and variable")
            for col in columns:
                if str(data[col].dtypes) == 'object':
                    st.text(
                        "Can't display the distribution plot of a categorical variable")
                else:
                    fig, ax = plt.subplots()
                    fig = plt.figure(figsize=(16, 9))
                    ax = sns.distplot(
                        data[data['Exited'] == 1][col], label="Exited")
                    ax = sns.distplot(
                        data[data['Exited'] == 0][col], label="Stayed")
                    plt.legend()
                    st.pyplot(fig)


def target_info(data, target):
    st.text('Value Counts By Target/Class')
    st.write(data[target].value_counts(normalize=True))
    st.write(data.iloc[:, -1].value_counts().plot.pie())

    fig = go.Figure(
        data=[go.Pie(labels=['Stayed', 'Exited'], values=data[target].value_counts())])

    fig.update_layout(title='Statistic of '+target, title_x=0.5, font_size=20)
    st.plotly_chart(fig)

    return data[target].value_counts(normalize=True)


def core(data, features, target, model, cv, length):

    trainset, testset = train_test_split(data, train_size=length)
    X_train, y_train = preprocessing(trainset)
    "Train size", y_train.value_counts()
    X_test, y_test = preprocessing(testset)
    "Test size", y_test.value_counts()

    evaluation(model, X_train, y_train, X_test, y_test, cv)

    predictions = model.predict(X_test)
    predictions_p = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f_score = f1_score(y_test, predictions, average="macro")
    p = precision_score(y_test, predictions, average="macro")
    r = recall_score(y_test, predictions, average="macro")
    ras = roc_auc_score(y_test, predictions_p[:, 1])
    accuracy_cv = 0
    if cv > 0:
        scores = cross_validate(model, data[features], data[target], cv=cv)
        accuracy_cv = np.mean(scores["test_score"])
    return predictions, predictions_p, accuracy, f_score, p, r, ras, accuracy_cv, y_test, X_test


def view(data, target, length, predictions, predictions_p, y_test):
    data_t = pd.DataFrame({"actual": y_test,
                           "predictions": predictions,
                           "predictions_proba": predictions_p[:, 1]})
    st.write(data_t)
    st.markdown("""
            <h6 style="font-size: 10px;">The column "predictions_proba" allows to determine the probability of success of the predicted value compared to 1.</h6>
            """,
                unsafe_allow_html=True)

    labels = ['actual_1', 'predictions_1', 'actual_0', 'predictions_0']
    values = [len(data_t.loc[data_t["actual"] == 1, "actual"]), len(data_t.loc[data_t["predictions"] == 1, "predictions"]),
              len(data_t.loc[data_t["actual"] == 0, "actual"]), len(data_t.loc[data_t["predictions"] == 0, "predictions"])]

    fig = px.bar(x=labels, y=values, width=620, height=420,
                 title="Actual and Predicted values of 0 and 1")
    fig.update_xaxes(title_text='values')
    fig.update_yaxes(title_text='number of values ​​present')
    st.plotly_chart(fig)
    return data_t


def main_content():
    st.markdown("""
        <h1 style="font-size: 50px; color:#DE781F" >Churn Prediction App</h1>
        """, unsafe_allow_html=True)
    # st.markdown("""
    #     Hello world :smiley:. You can download the project just here -->
    #     """)

    st.markdown("""
        Hello world :smiley:. You can see the project here--> <a href="https://github.com/badou11/streamlit_for_churn"/> Link</a>
        """,
                unsafe_allow_html=True)

    st.sidebar.markdown("""
            <h2 style="font-size: 15px;">Navigation</h2>
            """,
                        unsafe_allow_html=True)

    # separateur = st.sidebar.selectbox("Choose a separator", [',', ';'])
    uploaded = st.sidebar.file_uploader("upload", type='csv')

    if uploaded:
        data = load_data(uploaded)
        st.sidebar.write(data.shape)
        if data.shape[0] > 5000:
            reducer = st.sidebar.slider(
                "Randomly reduce data size %", min_value=0.2, max_value=0.9, value=0.5)
            reduction = data.shape[0]*reducer
            data = data.sample(int(reduction))
            st.sidebar.write(data.shape)
        st.sidebar.markdown("""
            <h2 style="font-size: 15px;">Frame</h2>
            """,
                            unsafe_allow_html=True)

        if st.sidebar.button('Display Dataframe'):
            "Raw Data", data.head(10)

        if st.sidebar.button('Some Statistics'):
            st.write(data.describe())

        target = st.sidebar.selectbox(
            'Choose the Target Variable : ', data.columns)
        if len(data[target].unique()) > 2:
            st.sidebar.warning("This variable have too much unique value")
            good_target = False
        elif data.dtypes[target] == 'object':
            st.sidebar.write(data[target].unique())
            st.sidebar.write(
                "This target Variable don't have numeric variable. Let's change it:")
            input1 = st.sidebar.text_input(
                f"Change {data[target].unique()[0]} into : ")
            input2 = st.sidebar.text_input(
                f"Change {data[target].unique()[1]} into : ")
            if st.sidebar.button("submit"):
                data[target] = data[target].map(
                    {data[target].unique()[0]: int(input1), data[target].unique()[1]: int(input2)})

                st.write(data)
                target_balance = target_info(data, target)
                good_target = True
        else:
            st.sidebar.info("We are good to go :smiley:")
            target_balance = target_info(data, target)
            good_target = True
        try:
            if target_balance[0] > 2*target_balance[1]:
                st.sidebar.write("this dataset is unbalanced")
                smote = st.sidebar.radio("smote method ?", ["Yes", "No"])
        except:
            pass

        st.sidebar.markdown("""
            <h2 style="font-size: 15px;">Visualizing</h2>
            """,
                            unsafe_allow_html=True)
        type_of_plot = st.sidebar.selectbox("Select a type of plot", [
                                            "Distribution", "Bar", "Histogram", "Boxplot", "Scatter", "Countplot"])
        if type_of_plot == "Histogram":
            bins = st.sidebar.number_input("Enter bins number : ")
            selected_columns_names = st.sidebar.selectbox(
                "Select a colomn", data.columns.tolist())

        elif type_of_plot == 'Countplot':
            selected_columns_names = st.sidebar.selectbox(
                "Select one column :", data.select_dtypes('object').columns)

        else:
            selected_columns_names = st.sidebar.multiselect(
                "Select columns", data.columns.tolist())

        if st.sidebar.button('Generate Plot'):
            st.success(
                f"Generating {type_of_plot} for {selected_columns_names}")
            customized_plot(type_of_plot, selected_columns_names,
                            data, target, bins=0)

        st.sidebar.markdown("""
            <h2 style="font-size: 15px;">Preprocessing</h2>
            """,
                            unsafe_allow_html=True)

        if st.sidebar.checkbox("Check null values"):
            st.write(data.isna().sum())
            null_vals = [i for i in data.isna().sum()]
            if np.sum(null_vals) != 0:
                st.write(
                    f"There is {np.sum(null_vals)} variable with null values")
                choice = st.sidebar.selectbox("How do you want to remove NaN values?", [
                                              'Dropna', 'Replace by Mean', 'Drop Columns with NaN'])
                missing_val_count_by_column = (data.isnull().sum())
                col_with_NaN = missing_val_count_by_column[missing_val_count_by_column > 0].index.to_list(
                )

                deal_with_NaN(data, col_with_NaN)
            else:
                st.write("Hum !! You are Lucky :smiley:")

        features = st.sidebar.multiselect(
            "Features", data.drop(target, axis=1).columns)

        if features:
            data = data[features + [target]]

            cat_variable = data.select_dtypes(
                'object').columns.to_list()
            if len(cat_variable) != 0:
                st.sidebar.write(f"{cat_variable} are categorical data")
                choice = st.sidebar.selectbox(f"Would you like to create dummies for them ?", [
                                              'Choose an options', 'OneHotEncoding', 'LabelEncoding'])

                if choice == 'OneHotEncoding':
                    try:
                        data = pd.get_dummies(
                            data=data, columns=cat_variable, drop_first=True)
                        st.write(data)
                    except:
                        st.sidebar.write('Choose only one option')
                elif choice == 'LabelEncoding':
                    try:
                        encoder = LabelEncoder()
                        for col in cat_variable:
                            data[col] = encoder.fit_transform(data[col])
                        st.write(data)
                    except:
                        st.sidebar.write('Choose only one option')
                else:
                    st.sidebar.warning("You have to choose an option")

        st.sidebar.markdown("""
            <h2 style="font-size: 15px;">Modeling</h2>
            """,
                            unsafe_allow_html=True)
        length = st.sidebar.slider(
            "Train size", min_value=0.2, max_value=0.9, value=0.8)

        cv = st.sidebar.selectbox(
            "Cross Validation on the train",
            [0, 5, 10, 15, 20])

        model = st.sidebar.selectbox(
            "Which model do you like!",
            ["Random Forest",
             "KnnClassifier",
             "Logistic Regression",
             "SgdClassifier",
             "Decision Tree",
             "SVClassification",
             "XGBoostClassifier"
             ])
        if model == "Decision Tree":
            params = ["criterion", "max_depth", "max_features",
                      "min_samples_leaf", "min_samples_split"]
            check_param = [st.sidebar.checkbox(
                param, key=param) for param in params]
            criterion, max_depth, max_features, min_samples_leaf, min_samples_split = "gini", None, None, 1, 2
            for p in range(len(params)):
                if check_param[p] and params[p] == "criterion":
                    criterion = st.sidebar.selectbox(
                        "enter criterion value",
                        ["gini", "entropy"]
                    )
                if check_param[p] and params[p] == "max_depth":
                    max_depth = st.sidebar.selectbox(
                        "enter max_depth value",
                        [None, 2, 5, 10, 15]
                    )
                if check_param[p] and params[p] == "max_features":
                    max_features = st.sidebar.selectbox(
                        "enter max_features value",
                        [None, "auto", "sqrt", "log2"]
                    )
                if check_param[p] and params[p] == "min_samples_leaf":
                    min_samples_leaf = st.sidebar.selectbox(
                        "enter min_samples_leaf value",
                        [1, 5, 8, 12]
                    )
                if check_param[p] and params[p] == "min_samples_split":
                    min_samples_split = st.sidebar.selectbox(
                        "enter min_samples_split value",
                        [2, 3, 5, 8]
                    )
            if st.sidebar.button("Predicting"):
                dt = DecisionTreeClassifier(random_state=0, criterion=criterion, max_depth=max_depth,
                                            max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
                if not features:
                    st.write("You have to choose some features for training")
                elif good_target == False:
                    st.write("Choose an appropriete target variable")
                else:
                    predictions, predictions_p, accuracy, f_score, p, r, ras, accuracy_cv, y_test, X_test = core(
                        data, features, target, dt, cv, length)
                    data_t = view(data, target, length,
                                  predictions, predictions_p, y_test)
                    tab = pd.DataFrame({"accuracy": [accuracy], "f1_score": [f_score],
                                        "precision_score": [p], "recall_score": [p],
                                        "roc_auc_score": [ras], "accuracy_cross_validation": [accuracy_cv]})
                    tab.index = [""] * len(tab)
                    st.markdown("""
                    <h2 style="font-size: 15px; text-decoration-line: underline;">Differents metrics</h2>
                    """,
                                unsafe_allow_html=True)

                    st.table(tab)

                    st.markdown("""
                    <h2 style="font-size: 15px; text-decoration-line: underline;">Calcul of your retention and churn rate</h2>
                    """,
                                unsafe_allow_html=True)
                    retention = (
                        len(data_t.loc[data_t["predictions"] == 0, "predictions"])/len(data_t))*100
                    churn = (
                        len(data_t.loc[data_t["predictions"] == 1, "predictions"])/len(data_t))*100
                    st.write("Retention rate: "+str(retention)+"%")
                    st.write("Churn rate: "+str(churn)+"%")

                    st.sidebar.markdown(download_link(
                        pd.concat([X_test, pd.DataFrame({"Predictions": predictions})], axis=1), "result.csv", "Download predicting results"), unsafe_allow_html=True)

        if model == "Random Forest":
            params = ["n_estimators", "criterion", "max_depth",
                      "max_features", "min_samples_leaf", "min_samples_split"]
            check_param = [st.sidebar.checkbox(
                param, key=param) for param in params]
            n_estimators, criterion, max_depth, max_features, min_samples_leaf, min_samples_split = 100, "gini", None, None, 1, 2
            for p in range(len(params)):
                if check_param[p] and params[p] == "n_estimators":
                    n_estimators = st.sidebar.selectbox(
                        "enter n_estimators value",
                        [100, 4, 6, 9]
                    )
                if check_param[p] and params[p] == "criterion":
                    criterion = st.sidebar.selectbox(
                        "enter criterion value",
                        ["gini", "entropy"]
                    )
                if check_param[p] and params[p] == "max_depth":
                    max_depth = st.sidebar.selectbox(
                        "enter max_depth value",
                        [None, 2, 5, 10, 15]
                    )
                if check_param[p] and params[p] == "max_features":
                    max_features = st.sidebar.selectbox(
                        "enter max_features value",
                        [None, "auto", "sqrt", "log2"]
                    )
                if check_param[p] and params[p] == "min_samples_leaf":
                    min_samples_leaf = st.sidebar.selectbox(
                        "enter min_samples_leaf value",
                        [1, 5, 8, 12]
                    )
                if check_param[p] and params[p] == "min_samples_split":
                    min_samples_split = st.sidebar.selectbox(
                        "enter min_samples_split value",
                        [2, 3, 5, 8]
                    )
            if st.sidebar.button("Predicting"):
                rf = RandomForestClassifier(random_state=0, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                            max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
                if not features:
                    st.write("You have to choose some features for training")
                elif good_target == False:
                    st.write("Choose an appropriete target variable")
                else:
                    predictions, predictions_p, accuracy, f_score, p, r, ras, accuracy_cv, y_test, X_test = core(
                        data, features, target, rf, cv, length)
                    data_t = view(data, target, length,
                                  predictions, predictions_p, y_test)
                    tab = pd.DataFrame({"accuracy": [accuracy], "f1_score": [f_score],
                                        "precision_score": [p], "recall_score": [p],
                                        "roc_auc_score": [ras], "accuracy_cross_validation": [accuracy_cv]})
                    tab.index = [""] * len(tab)
                    st.markdown("""
                    <h2 style="font-size: 15px; text-decoration-line: underline;">Differents metrics</h2>
                    """,
                                unsafe_allow_html=True)

                    st.table(tab)

                    st.markdown("""
                    <h2 style="font-size: 15px; text-decoration-line: underline;">Calcul of your retention and churn rate</h2>
                    """,
                                unsafe_allow_html=True)
                    retention = (
                        len(data_t.loc[data_t["predictions"] == 0, "predictions"])/len(data_t))*100
                    churn = (
                        len(data_t.loc[data_t["predictions"] == 1, "predictions"])/len(data_t))*100
                    st.write("Retention rate: "+str(retention)+"%")
                    st.write("Churn rate: "+str(churn)+"%")

                    # st.sidebar.markdown(download_link(
                    #     data_t, "result.csv", "Download predicting results"), unsafe_allow_html=True)

                    st.sidebar.markdown(download_link(
                        pd.concat([X_test, pd.DataFrame({"Predictions": predictions})], axis=1), "result.csv", "Download predicting results"), unsafe_allow_html=True)

        if model == "KnnClassifier":
            params = ["n_neighbors", "weights", "algorithm"]
            check_param = [st.sidebar.checkbox(
                param, key=param) for param in params]
            n_neighbors, weights, algorithm = 5, "uniform", "auto"
            for p in range(len(params)):
                if check_param[p] and params[p] == "n_neighbors":
                    n_neighbors = st.sidebar.selectbox(
                        "enter n_neighbors value",
                        [5, 10, 15, 20, 25]
                    )
                if check_param[p] and params[p] == "weights":
                    weights = st.sidebar.selectbox(
                        "enter weights value",
                        ["uniform", "distance"]
                    )
                if check_param[p] and params[p] == "algorithm":
                    algorithm = st.sidebar.selectbox(
                        "enter algorithm value",
                        ["auto", "ball_tree", "kd_tree", "brute"]
                    )
            if st.sidebar.button("Predicting"):
                knn = KNeighborsClassifier(
                    n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
                if not features:
                    st.write("You have to choose some features for training")
                elif good_target == False:
                    st.write("Choose an appropriete target variable")
                else:
                    predictions, predictions_p, accuracy, f_score, p, r, ras, accuracy_cv, y_test, X_test = core(
                        data, features, target, knn, cv, length)
                    data_t = view(data, target, length,
                                  predictions, predictions_p, y_test)
                    tab = pd.DataFrame({"accuracy": [accuracy], "f1_score": [f_score],
                                        "precision_score": [p], "recall_score": [p],
                                        "roc_auc_score": [ras], "accuracy_cross_validation": [accuracy_cv]})
                    tab.index = [""] * len(tab)
                    st.markdown("""
                    <h2 style="font-size: 15px; text-decoration-line: underline;">Differents metrics</h2>
                    """,
                                unsafe_allow_html=True)

                    st.table(tab)

                    st.markdown("""
                    <h2 style="font-size: 15px; text-decoration-line: underline;">Calcul of your retention and churn rate</h2>
                    """,
                                unsafe_allow_html=True)
                    retention = (
                        len(data_t.loc[data_t["predictions"] == 0, "predictions"])/len(data_t))*100
                    churn = (
                        len(data_t.loc[data_t["predictions"] == 1, "predictions"])/len(data_t))*100
                    st.write("Retention rate: "+str(retention)+"%")
                    st.write("Churn rate: "+str(churn)+"%")

                    # st.sidebar.markdown(download_link(
                    #     data_t, "result.csv", "Download predicting results"), unsafe_allow_html=True)

                    st.sidebar.markdown(download_link(
                        pd.concat([X_test, pd.DataFrame({"Predictions": predictions})], axis=1), "result.csv", "Download predicting results"), unsafe_allow_html=True)

        if model == "Logistic Regression":
            params = ["penalty", "solver"]
            check_param = [st.sidebar.checkbox(
                param, key=param) for param in params]
            penalty, solver = "l2", "lbfgs"
            for p in range(len(params)):
                if check_param[p] and params[p] == "penalty":
                    penalty = st.sidebar.selectbox(
                        "enter penalty value",
                        ["l2", "l1", "elasticnet", "none"]
                    )
                if check_param[p] and params[p] == "solver":
                    solver = st.sidebar.selectbox(
                        "enter solver value",
                        ["lbfgs", "newton-cg", "liblinear", "sag", "saga"]
                    )
            if st.sidebar.button("Predicting"):
                lr = LogisticRegression(
                    random_state=0, penalty=penalty, solver=solver)
                if not features:
                    st.write("You have to choose some features for training")
                elif good_target == False:
                    st.write("Choose an appropriete target variable")
                else:
                    predictions, predictions_p, accuracy, f_score, p, r, ras, accuracy_cv, y_test, X_test = core(
                        data, features, target, lr, cv, length)
                    data_t = view(data, target, length,
                                  predictions, predictions_p, y_test)
                    tab = pd.DataFrame({"accuracy": [accuracy], "f1_score": [f_score],
                                        "precision_score": [p], "recall_score": [p],
                                        "roc_auc_score": [ras], "accuracy_cross_validation": [accuracy_cv]})
                    tab.index = [""] * len(tab)
                    st.markdown("""
                    <h2 style="font-size: 15px; text-decoration-line: underline;">Differents metrics</h2>
                    """,
                                unsafe_allow_html=True)

                    st.table(tab)

                    st.markdown("""
                    <h2 style="font-size: 15px; text-decoration-line: underline;">Calcul of your retention and churn rate</h2>
                    """,
                                unsafe_allow_html=True)
                    retention = (
                        len(data_t.loc[data_t["predictions"] == 0, "predictions"])/len(data_t))*100
                    churn = (
                        len(data_t.loc[data_t["predictions"] == 1, "predictions"])/len(data_t))*100
                    st.write("Retention rate: "+str(retention)+"%")
                    st.write("Churn rate: "+str(churn)+"%")
                    # st.sidebar.markdown(download_link(
                    #     data_t, "result.csv", "Download predicting results"), unsafe_allow_html=True)

                    st.sidebar.markdown(download_link(
                        pd.concat([X_test, pd.DataFrame({"Predictions": predictions})], axis=1), "result.csv", "Download predicting results"), unsafe_allow_html=True)

        if model == "SgdClassifier":
            params = ["loss", "penalty"]
            check_param = [st.sidebar.checkbox(
                param, key=param) for param in params]
            loss, penalty = "hinge", "l2"
            for p in range(len(params)):
                if check_param[p] and params[p] == "loss":
                    loss = st.sidebar.selectbox(
                        "enter hinge value",
                        ["hinge", "log", "modified_huber",
                         "squared_hinge", "perceptron"]
                    )
                if check_param[p] and params[p] == "penalty":
                    penalty = st.sidebar.selectbox(
                        "enter penalty value",
                        ["l2", "l1", "elasticnet"]
                    )
            if st.sidebar.button("Predicting"):
                sc = SGDClassifier(random_state=0, loss=loss, penalty=penalty)
                if not features:
                    st.write("You have to choose some features for training")
                elif good_target == False:
                    st.write("Choose an appropriete target variable")
                else:
                    predictions, predictions_p, accuracy, f_score, p, r, ras, accuracy_cv, y_test, X_test = core(
                        data, features, target, sc, cv, length)
                    data_t = view(data, target, length,
                                  predictions, predictions_p, y_test)
                    tab = pd.DataFrame({"accuracy": [accuracy], "f1_score": [f_score],
                                        "precision_score": [p], "recall_score": [p],
                                        "roc_auc_score": [ras], "accuracy_cross_validation": [accuracy_cv]})
                    tab.index = [""] * len(tab)
                    st.markdown("""
                    <h2 style="font-size: 15px; text-decoration-line: underline;">Differents metrics</h2>
                    """,
                                unsafe_allow_html=True)

                    st.table(tab)

                    st.markdown("""
                    <h2 style="font-size: 15px; text-decoration-line: underline;">Calcul of your retention and churn rate</h2>
                    """,
                                unsafe_allow_html=True)
                    retention = (
                        len(data_t.loc[data_t["predictions"] == 0, "predictions"])/len(data_t))*100
                    churn = (
                        len(data_t.loc[data_t["predictions"] == 1, "predictions"])/len(data_t))*100
                    st.write("Retention rate: "+str(retention)+"%")
                    st.write("Churn rate: "+str(churn)+"%")

                    st.sidebar.markdown(download_link(
                        pd.concat([X_test, pd.DataFrame({"Predictions": predictions})], axis=1), "result.csv", "Download predicting results"), unsafe_allow_html=True)

        if model == "SVClassification":
            params = ["kernel", "degree"]
            check_param = [st.sidebar.checkbox(
                param, key=param) for param in params]
            kernel, degree = "rbf", 3
            for p in range(len(params)):
                if check_param[p] and params[p] == "kernel":
                    kernel = st.sidebar.selectbox(
                        "enter kernel value",
                        ["rbf", "poly", "sigmoid", "precomputed"]
                    )
                if check_param[p] and params[p] == "degree":
                    degree = st.sidebar.selectbox(
                        "enter degree value",
                        [3, 6, 9]
                    )
            if st.sidebar.button("Predicting"):
                sv = SVC(random_state=0, kernel=kernel,
                         degree=degree, probability=True)
                if not features:
                    st.write("You have to choose some features for training")
                elif good_target == False:
                    st.write("Choose an appropriete target variable")
                else:
                    predictions, predictions_p, accuracy, f_score, p, r, ras, accuracy_cv, y_test, X_test = core(
                        data, features, target, sv, cv, length)
                    data_t = view(data, target, length,
                                  predictions, predictions_p, y_test)
                    tab = pd.DataFrame({"accuracy": [accuracy], "f1_score": [f_score],
                                        "precision_score": [p], "recall_score": [p],
                                        "roc_auc_score": [ras], "accuracy_cross_validation": [accuracy_cv]})
                    tab.index = [""] * len(tab)
                    st.markdown("""
                    <h2 style="font-size: 15px; text-decoration-line: underline;">Differents metrics</h2>
                    """,
                                unsafe_allow_html=True)

                    st.table(tab)

                    st.markdown("""
                    <h2 style="font-size: 15px; text-decoration-line: underline;">Calcul of your retention and churn rate</h2>
                    """,
                                unsafe_allow_html=True)
                    retention = (
                        len(data_t.loc[data_t["predictions"] == 0, "predictions"])/len(data_t))*100
                    churn = (
                        len(data_t.loc[data_t["predictions"] == 1, "predictions"])/len(data_t))*100
                    st.write("Retention rate: "+str(retention)+"%")
                    st.write("Churn rate: "+str(churn)+"%")

                    st.sidebar.markdown(download_link(
                        pd.concat([X_test, pd.DataFrame({"Predictions": predictions})], axis=1), "result.csv", "Download predicting results"), unsafe_allow_html=True)


def deal_with_NaN(data, col_with_NaN):
    if choice == "Dropna":
        return data.dropna(axis=0)

    if choice == "Replace by Mean":
        imputer = SimpleImputer(strategy='mean')
        Imputed_data = pd.DataFrame(imputer.fit_transform(data))
        Imputed_data = data.columns
        return Imputed_data

    if choice == "Drop Columns with NaN":
        return data.drop(columns=col_with_NaN)


def corr_matrix(data):
    fig, ax = plt.subplots()
    ax = sns.heatmap(data.corr(), annot=True, cbar=False)
    st.pyplot(fig)


def preprocessing(data):
    X = data.drop('Exited', axis=1)
    y = data.Exited

    return X, y


def evaluation(model, X_train, y_train, X_test, y_test, cv):
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    st.write("Correlation Matrix")
    st.write(confusion_matrix(y_test, ypred))
    st.write("Classification Repport")
    st.write(classification_report(y_test, ypred))

    N, train_scores, test_scores = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.1, 1, 10),
                                                  cv=10)

    fig = plt.subplots()
    fig = plt.figure(figsize=(12, 8))
    ax = plt.plot(N, train_scores.mean(axis=1), label='train score')
    ax = plt.plot(N, test_scores.mean(axis=1), label='test score')
    ax = plt.title(
        "Learning curve for accuracy: This show us if the model overfit")
    plt.legend()
    st.pyplot(fig)

    return model


def main():
    """Common Machine Learning EDA"""

    main_content()


if __name__ == '__main__':
    main()

# def highlight_max(data):
#     '''
#     highlight the maximum in a Series yellow.
#     '''
#     if data.iloc[:, 0] == data.iloc[:, 1]

#         return ['background-color: yellow' if v else '' for v in is_max]
