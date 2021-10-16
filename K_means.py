import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import  Image
from io import BytesIO

# Custom giao diện web
st.set_page_config(
    page_title="K-mean Clustering",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
)
image = Image.open("hello.png")
st.image(image)
# Tạo title
st.title("K-means application website")
st.markdown("""
This is our project on how to visualize data using Kmeans-Clustering
* **Used library**: streamlit, pandas, matplotlib, sklearn.cluster, sklearn.preprocessing
* **Data source**: [archive.ics.uci.edu](https://archive.ics.uci.edu/), [kaggle.com](https://www.kaggle.com/)
""")
# Tạo thanh side bar
st.sidebar.markdown("""
# **Use Input Parameters**
""")
# Tạo phần upload file

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=['csv'])
if uploaded_file is not None:
    st.title("K-mean Clustering")
    # Gán file vừa đọc vào biến data
    data = pd.read_csv(uploaded_file, nrows=500)
    max_feature = data.shape[1] - 1
    # Tạo thanh slider ứng với 2 feature 
    st.sidebar.write("**Choose input feature**:")
    form = st.sidebar.form("form_2")
    f1 = form.slider("First Feature", 0, max_feature)
    f2 = form.slider("Second Feature", 0, max_feature,1)
    check_2 = form.form_submit_button("Submit")
    # Gán X với giá trị của 2 feature
    X = data.iloc[:, [f1, f2]].values
    # Tạo hàm chuẩn hóa dữ liệu
    def scaler_data(X):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    # Tạo thanh slider để chọn input cho K và thanh checkbox để gọi hàm chuẩn hóa
    with st.sidebar.form("my_form"):
        k = st.slider("Choose K",1,8)
        check = st.checkbox("Standardscaler")
        if check:
            X = scaler_data(X)
            st.success("Data was updated")
        st.form_submit_button("Submit")
    # Khởi tạo khung data
    'Data', data
    # Tạo X
    st.markdown("""
        # Graph to determine K
        """)
    try:
        # Vẽ đồ thị cẳng tay để tìm K
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(range(1, 11), wcss)
        # Tạo ra đồ thị Elbow và dataframe của 2 input feature
        col1, col2 = st.columns([2, 2])
        with col1:
            st.subheader("Elbow Graph")
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)
        with col2:
            st.subheader("Data by feature")
            data.iloc[:,[f1,f2]]
        # Train mô hình Kmeans
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(X)
        st.subheader("**Cluster Graph**")
        # Hàm vẽ đồ thị
        def Kmean(k):
            color = ['red', 'blue', 'green', 'cyan', 'magenta','orange', 'purple', 'gray']
            fig, ax = plt.subplots(figsize=(8, 8))
            for i in range(0, k):
                ax.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=color[i], label='Cluster %d' % (i))
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black',
                        marker="*", label='Centroids')
            ax.set(xlabel=data.columns.values[f1],
                    ylabel=data.columns.values[f2])
            ax.legend()
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)
        Kmean(k)
        st.markdown("""
        ### After cluster the data we created a new data frame include old data + new column **[Cluster]**
        * You can dowload the dataframes below
        """)
        # Tạo dataframe với column mới: "Cụm"
        data["Cluster"] = y_kmeans
        "Clustered Data", data
        # Tạo nút dowload file data(csv)
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(data)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='Data_Cluster.csv',
            mime='text/csv',
        )

        st.markdown("""
        * Choose the clustering you want
        """)
        # Tạo dataframe khi đã được Lọc với cụm
        c = []
        for i in range(0,k):
            c.append(i)
        new_data = data
        with st.form("form_3"):
            c_choose = st.multiselect("Cluster",c)
            st.write(c_choose)
            check_3 = st.form_submit_button("Submit")
            if check_3:
                new_data = data.loc[data.Cluster.isin(c_choose)]
        "Filtered Data", new_data
        # Tạo nút dowload file data(csv)
        if check_3:
            csv2 = convert_df(new_data)
            st.download_button(
                label="Download data as CSV",
                data=csv2,
                file_name='Data_Cluster_Filter.csv',
                mime='text/csv',
            )
    except (ValueError,st.StreamlitAPIException):
            st.warning("Select only input features that are number array and 2 features must be different")
# Khi chưa có file được upload thì sẽ hiển thị cửa sổ cảnh báo
else:
    st.warning("Please Upload a file")
