import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import warnings
import plotly.graph_objects as go # Import Plotly Go untuk tab perbandingan

warnings.filterwarnings('ignore')

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis Pola Ekspor-Impor di Indonesia",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# --- Definisi Warna ---
color_map = {
    'Ekspor Rendah': '#581845', 'Ekspor Sedang': '#20B2AA', 'Ekspor Tinggi': '#FFD700',
    'Impor Rendah': '#581845', 'Impor Sedang': '#20B2AA', 'Impor Tinggi': '#FFD700'
}
color_order_ekspor = ['Ekspor Rendah', 'Ekspor Sedang', 'Ekspor Tinggi']
color_order_impor = ['Impor Rendah', 'Impor Sedang', 'Impor Tinggi']
palette_ekspor = [color_map.get(cat, '#808080') for cat in color_order_ekspor]
palette_impor = [color_map.get(cat, '#808080') for cat in color_order_impor]


# --- Path File ---
EKSPOR_FILE_PATH = "./Data_Ekspor_Tahun2012-2025.xlsx"
IMPOR_FILE_PATH = "./Data_Impor_Tahun2012-2025.xlsx"

# --- Fungsi Utility ---
@st.cache_data
def clean_data(df, cols_to_clean):
    """Membersihkan kolom numerik (string -> float)."""
    df_cleaned = df.copy()
    for col in cols_to_clean:
        if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    df_cleaned = df_cleaned.dropna(subset=cols_to_clean)
    return df_cleaned

@st.cache_data(show_spinner="Memproses data...")
def process_data(file_path, data_type):
    """Memuat, membersihkan, cluster, label, dan PCA data ekspor/impor."""
    is_ekspor = data_type == 'ekspor'
    sheet_name = "Sheet1" # Asumsi nama sheet

    # 1. Load & Clean
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        st.error(f"File tidak ditemukan: {file_path}")
        return None
    except Exception as e:
        st.error(f"Gagal memuat file {file_path}: {e}")
        return None

    cols_to_clean = (['Total', 'MIGAS', 'NON MIGAS', 'Agriculture', 'Industry', 'Mining', 'Others'] if is_ekspor
                     else ['Total', 'Consumption Goods', 'Raw Material Support', 'Capital Goods'])
    numeric_cols = cols_to_clean[1:] # Kolom untuk analisis (tanpa 'Total')

    df = clean_data(df, cols_to_clean)
    # Pastikan kolom numerik ada
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if not numeric_cols or df.empty:
        st.warning(f"Data {data_type} tidak valid atau kolom numerik tidak ditemukan setelah pembersihan.")
        return None

    # 2. Feature Engineering (Impor Saja) & Scaling
    if is_ekspor:
        feature_cols = numeric_cols
        data_to_scale = df[feature_cols]
    else: # Impor
        proporsi_cols = []
        df['Total_clean'] = df['Total'].replace(0, 1e-9)
        for col in numeric_cols:
            prop_col_name = f'Proporsi_{col}'
            df[prop_col_name] = (df[col] / df['Total_clean']) * 100
            df[prop_col_name] = df[prop_col_name].replace([np.inf, -np.inf], np.nan).fillna(0)
            proporsi_cols.append(prop_col_name)
        df = df.drop(columns=['Total_clean'])
        feature_cols = proporsi_cols # Cluster berdasarkan proporsi
        data_to_scale = df[feature_cols]

    if data_to_scale.empty:
        st.warning(f"Tidak ada data valid untuk scaling {data_type}.")
        return None

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_to_scale)

    # 3. K-Means Clustering
    if data_scaled.shape[0] < 3: # Butuh minimal 3 sampel untuk 3 cluster
        st.warning(f"Data {data_type} terlalu sedikit ({data_scaled.shape[0]} baris) untuk K=3.")
        return None

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(data_scaled)

    # 4. Pelabelan Cluster (berdasarkan nilai asli)
    label_prefix = "Ekspor" if is_ekspor else "Impor"
    summary_cols = numeric_cols # Gunakan nilai asli untuk pelabelan
    cluster_summary = df.groupby('Cluster')[summary_cols].mean()
    cluster_summary['Avg_Value'] = cluster_summary.mean(axis=1)
    cluster_summary = cluster_summary.sort_values(by='Avg_Value', ascending=False)
    ordered_clusters = cluster_summary.index.tolist()

    label_map = {}
    if len(ordered_clusters) >= 3:
        label_map = {ordered_clusters[0]: f'{label_prefix} Tinggi', ordered_clusters[1]: f'{label_prefix} Sedang', ordered_clusters[2]: f'{label_prefix} Rendah'}
    # (Tambahkan penanganan jika < 3 cluster jika perlu)
    df['Kategori'] = df['Cluster'].map(label_map)

    # 5. PCA (pada data_scaled)
    pca_cols = {}
    try:
        pca_2d = PCA(n_components=2, random_state=42)
        pca_data_2d = pca_2d.fit_transform(data_scaled)
        df['PCA1'] = pca_data_2d[:, 0]
        df['PCA2'] = pca_data_2d[:, 1]
        pca_cols['2D'] = ['PCA1', 'PCA2']
    except ValueError: pass # Lewati jika tidak bisa

    if not is_ekspor: # PCA 3D hanya untuk impor (berdasarkan proporsi)
       try:
           pca_3d = PCA(n_components=3, random_state=42)
           pca_data_3d = pca_3d.fit_transform(data_scaled)
           df['PCA3'] = pca_data_3d[:, 2] # Tambah PCA3
           pca_cols['3D'] = ['PCA1', 'PCA2', 'PCA3']
       except ValueError: pass

    # Ringkasan nilai asli per kategori
    summary_by_category = df.groupby('Kategori')[summary_cols].mean().round(2) if 'Kategori' in df.columns else pd.DataFrame()

    return {
        "df": df, "data_scaled": data_scaled, "numeric_cols": numeric_cols,
        "feature_cols": feature_cols, # Kolom yg di-cluster (proporsi u/ impor)
        "category_col": "Kategori", "cluster_col": "Cluster",
        "pca_cols": pca_cols, "summary_by_category": summary_by_category
    }

@st.cache_data(show_spinner="Menghitung metrik evaluasi...")
def get_eval_metrics(data_scaled, max_k=10):
    """Hitung inertia, silhouette, davies-bouldin."""
    # (Fungsi ini sama seperti sebelumnya, pastikan robust)
    inertia, sil_scores, db_scores = [], [], []
    k_limit = data_scaled.shape[0] if data_scaled is not None else 1
    K_range = range(1, min(max_k + 1, k_limit))
    K_range_eval = range(2, min(max_k + 1, k_limit))

    if data_scaled is None or k_limit < 2: return [], [], [], [], []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data_scaled)
        inertia.append(kmeans.inertia_)
    for n in K_range_eval:
        labels = KMeans(n_clusters=n, random_state=42, n_init=10).fit_predict(data_scaled)
        if len(np.unique(labels)) > 1:
            sil_scores.append(silhouette_score(data_scaled, labels))
            db_scores.append(davies_bouldin_score(data_scaled, labels))
        else:
            sil_scores.append(np.nan); db_scores.append(np.nan)
    return list(K_range), inertia, list(K_range_eval)[:len(sil_scores)], sil_scores, db_scores

# --- Fungsi Plotting (Matplotlib & Seaborn) ---
def plot_cluster_countplot(df, category_col, order, palette):
    """Tampilkan countplot distribusi cluster."""
    if category_col not in df.columns: return
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=category_col, data=df, order=order, palette=palette, ax=ax)
    ax.set_title("Distribusi Data per Cluster"); ax.set_xlabel("Kategori")
    ax.tick_params(axis='x', rotation=30)
    st.pyplot(fig); plt.clf()

def plot_cluster_comparison_bar(summary_by_category, order, colors):
    """Tampilkan bar plot perbandingan rata-rata cluster."""
    if summary_by_category.empty: return
    try: summary = summary_by_category.reindex(order) # Urutkan
    except KeyError: summary = summary_by_category # Biarkan jika ada yg hilang
    fig, ax = plt.subplots(figsize=(10, 5))
    summary.T.plot(kind='bar', color=[colors.get(cat, '#808080') for cat in summary.index], ax=ax)
    ax.set_title("Perbandingan Rata-Rata per Cluster (Nilai Asli)")
    ax.set_ylabel("Rata-rata (Juta USD)"); ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Kategori')
    st.pyplot(fig); plt.clf()

def plot_yearly_distribution_stackedbar(df, category_col, order, colors):
    """Tampilkan stacked bar chart distribusi kategori per tahun."""
    if category_col not in df.columns or 'Tahun' not in df.columns: return
    period_analysis = df.groupby(['Tahun', category_col]).size().unstack(fill_value=0)
    cols_present = [col for col in order if col in period_analysis.columns] # Kolom yg ada saja
    period_analysis = period_analysis[cols_present]
    fig, ax = plt.subplots(figsize=(10, 5))
    period_analysis.plot(kind='bar', stacked=True, color=[colors.get(cat, '#808080') for cat in cols_present], ax=ax)
    ax.set_title('Distribusi Kategori per Tahun'); ax.set_xlabel('Tahun'); ax.set_ylabel('Jumlah Bulan')
    ax.legend(title='Kategori', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig); plt.clf()

def plot_yearly_detail_bar(df_year, numeric_cols, year, title_suffix):
    """Tampilkan bar plot rata-rata sektor/kategori untuk 1 tahun."""
    if df_year.empty or not numeric_cols: return None # Kembalikan None jika tidak ada data
    avg_data = df_year[numeric_cols].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_data.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_title(f"Rata-Rata {title_suffix} Tahun {year}")
    ax.set_xlabel(title_suffix); ax.set_ylabel("Rata-Rata Nilai (Juta USD)")
    ax.tick_params(axis='x', rotation=45); ax.grid(axis='y', linestyle='--', alpha=0.7)
    return fig # Kembalikan figure object

def plot_pca_2d(df, category_col, pca_cols, order, palette):
    """Tampilkan scatter plot PCA 2D."""
    pca1, pca2 = pca_cols
    if category_col not in df.columns or pca1 not in df.columns or df[pca1].isna().all(): return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x=pca1, y=pca2, hue=category_col, palette=palette, hue_order=order, s=60, ax=ax)
    ax.set_title('Visualisasi Cluster (PCA 2D)'); ax.legend(title='Kategori', fontsize='small')
    st.pyplot(fig); plt.clf()

def plot_pca_3d(df, category_col, pca_cols, cluster_col, color_map, title, z_col=None, z_label="Nilai"):
    """Tampilkan scatter plot PCA 3D."""
    pca1, pca2, pca3 = pca_cols
    req_cols = [category_col, cluster_col, pca1, pca2] + ([pca3] if z_col is None else [z_col])
    if not all(c in df.columns for c in req_cols) or df[pca1].isna().all(): return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = df[category_col].map(color_map).fillna('#808080')
    z_values = df[pca3] if z_col is None else df[z_col]
    z_label_final = "PCA3" if z_col is None else z_label

    scatter = ax.scatter(df[pca1], df[pca2], z_values, c=colors, s=40)
    ax.set_xlabel('PCA1'); ax.set_ylabel('PCA2'); ax.set_zlabel(z_label_final)
    ax.set_title(title)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=cat, markerfacecolor=col, markersize=8)
               for cat, col in color_map.items() if cat.startswith(category_col.split(" ")[0])] # Ekspor atau Impor
    ax.legend(handles=handles, title="Kategori", fontsize='small', bbox_to_anchor=(1.15, 1))
    st.pyplot(fig); plt.clf()

# --- Muat Data Awal ---
data_ekspor = process_data(EKSPOR_FILE_PATH, 'ekspor')
data_impor = process_data(IMPOR_FILE_PATH, 'impor')

# --- UI Utama ---
if data_ekspor or data_impor:
    tab_ekspor, tab_impor, tab_perbandingan = st.tabs(["📤 Analisis Ekspor", "📥 Analisis Impor", "📊 Perbandingan"])

    # === TAB EKSPOR ===
    with tab_ekspor:
        if data_ekspor:
            st.header("1. Analisis Pola Ekspor di Indonesia"); st.divider()
            df_e = data_ekspor["df"] # Unpack data for easier access
            scaled_e = data_ekspor["data_scaled"]
            num_cols_e = data_ekspor["numeric_cols"]
            cat_col_e = data_ekspor["category_col"]
            cluster_col_e = data_ekspor["cluster_col"]
            pca_cols_e = data_ekspor["pca_cols"]
            summary_cat_e = data_ekspor["summary_by_category"]

            st.subheader("Hasil Clustering (K=3)")
            col1, col2 = st.columns([1, 2])
            with col1:
                plot_cluster_countplot(df_e, cat_col_e, color_order_ekspor, palette_ekspor)
                st.write("**Evaluasi Final (K=3)**")
                if len(df_e[cluster_col_e].unique()) > 1:
                    st.metric("Silhouette Score", f"{silhouette_score(scaled_e, df_e[cluster_col_e]):.3f}")
                    st.metric("Davies-Bouldin Index", f"{davies_bouldin_score(scaled_e, df_e[cluster_col_e]):.3f}")
            with col2:
                plot_cluster_comparison_bar(summary_cat_e, color_order_ekspor, color_map)

            st.divider(); st.write("**Distribusi Kategori per Tahun**")
            plot_yearly_distribution_stackedbar(df_e, cat_col_e, color_order_ekspor, color_map)

            st.divider(); st.subheader("Analisis Mendalam per Tahun")
            years_e = sorted(df_e['Tahun'].unique(), reverse=True)
            year_e = st.selectbox("Pilih Tahun Ekspor:", years_e, key='sel_year_e')
            df_year_e = df_e[df_e['Tahun'] == year_e]
            col1_yr, col2_yr = st.columns([2, 1])
            with col1_yr:
                fig_yr = plot_yearly_detail_bar(df_year_e, num_cols_e, year_e, "Sektor")
                if fig_yr: st.pyplot(fig_yr); plt.clf()
                else: st.warning(f"Tidak ada data ekspor untuk tahun {year_e}.")
            with col2_yr:
                if not df_year_e.empty:
                    st.write("**Distribusi Kategori**"); st.dataframe(df_year_e[cat_col_e].value_counts())
                    st.write("**Rata-rata Sektor**"); st.dataframe(df_year_e[num_cols_e].mean().round(2))
            if not df_year_e.empty:
                st.write("**Detail Analisis Data**")
                cols_show_yr = ['Bulan', cat_col_e, 'Total'] + num_cols_e
                st.dataframe(df_year_e[[c for c in cols_show_yr if c in df_year_e.columns]])


            st.divider(); st.subheader("Visualisasi Cluster PCA")
            col1_pca, col2_pca = st.columns(2)
            with col1_pca:
                if '2D' in pca_cols_e: plot_pca_2d(df_e, cat_col_e, pca_cols_e['2D'], color_order_ekspor, color_map)
            with col2_pca:
                # Plot 3D Ekspor: PCA1, PCA2, Total
                if '2D' in pca_cols_e and 'Total' in df_e.columns:
                    plot_pca_3d(df_e, cat_col_e, pca_cols_e['2D'] + ['DUMMY_PCA3'], # Butuh 3 nama kolom
                                cluster_col_e, color_map, "Cluster Ekspor 3D", z_col='Total', z_label="Total (Juta USD)")


            st.divider(); st.subheader("💾 Download Data Ekspor Hasil Analisis")
            csv_e = df_e.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Data Ekspor (.csv)", csv_e, f"hasil_ekspor_{pd.Timestamp.now():%Y%m%d_%H%M}.csv", "text/csv", key="dl_e")

        else:
            st.warning("Data ekspor tidak dapat dimuat atau diproses.")

    # === TAB IMPOR ===
    with tab_impor:
        if data_impor:
            st.header("2. Analisis Pola Impor di Indonesia"); st.divider()
            df_i = data_impor["df"]
            scaled_i = data_impor["data_scaled"]
            num_cols_i = data_impor["numeric_cols"] # Kolom nilai asli
            feature_cols_i = data_impor["feature_cols"] # Kolom proporsi (untuk judul plot evaluasi)
            cat_col_i = data_impor["category_col"]
            cluster_col_i = data_impor["cluster_col"]
            pca_cols_i = data_impor["pca_cols"]
            summary_cat_i = data_impor["summary_by_category"] # Summary nilai asli

            st.subheader("Hasil Clustering (K=3)")
            col1, col2 = st.columns([1, 2])
            with col1:
                plot_cluster_countplot(df_i, cat_col_i, color_order_impor, palette_impor)
                st.write("**Evaluasi Final (K=3 pada Proporsi)**")
                if len(df_i[cluster_col_i].unique()) > 1:
                    st.metric("Silhouette Score", f"{silhouette_score(scaled_i, df_i[cluster_col_i]):.3f}")
                    st.metric("Davies-Bouldin Index", f"{davies_bouldin_score(scaled_i, df_i[cluster_col_i]):.3f}")
            with col2:
                plot_cluster_comparison_bar(summary_cat_i, color_order_impor, color_map)

            st.divider(); st.write("**Distribusi Kategori per Tahun**")
            plot_yearly_distribution_stackedbar(df_i, cat_col_i, color_order_impor, color_map)

            st.divider(); st.subheader("Analisis Mendalam per Tahun")
            years_i = sorted(df_i['Tahun'].unique(), reverse=True)
            year_i = st.selectbox("Pilih Tahun Impor:", years_i, key='sel_year_i')
            df_year_i = df_i[df_i['Tahun'] == year_i]
            col1_yr_i, col2_yr_i = st.columns([2, 1])
            with col1_yr_i:
                fig_yr_i = plot_yearly_detail_bar(df_year_i, num_cols_i, year_i, "Kategori Impor")
                if fig_yr_i: st.pyplot(fig_yr_i); plt.clf()
                else: st.warning(f"Tidak ada data impor untuk tahun {year_i}.")
            with col2_yr_i:
                if not df_year_i.empty:
                    st.write("**Distribusi Kategori**"); st.dataframe(df_year_i[cat_col_i].value_counts())
                    st.write("**Rata-rata Kategori**"); st.dataframe(df_year_i[num_cols_i].mean().round(2))
            if not df_year_i.empty:
                st.write("**Detail Analisis Data**")
                cols_show_yr_i = ['Bulan', cat_col_i, 'Total'] + num_cols_i
                st.dataframe(df_year_i[[c for c in cols_show_yr_i if c in df_year_i.columns]])

            st.divider(); st.subheader("Visualisasi Cluster PCA (berdasarkan Proporsi)")
            col1_pca_i, col2_pca_i = st.columns(2)
            with col1_pca_i:
                if '2D' in pca_cols_i: plot_pca_2d(df_i, cat_col_i, pca_cols_i['2D'], color_order_impor, color_map)
            with col2_pca_i:
                if '3D' in pca_cols_i: plot_pca_3d(df_i, cat_col_i, pca_cols_i['3D'], cluster_col_i, color_map, "Cluster Impor 3D (PCA-Proporsi)")


            st.divider(); st.subheader("💾 Download Data Impor Hasil Analisis")
            csv_i = df_i.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Data Impor (.csv)", csv_i, f"hasil_impor_{pd.Timestamp.now():%Y%m%d_%H%M}.csv", "text/csv", key="dl_i")

        else:
            st.warning("Data impor tidak dapat dimuat atau diproses.")

    # === TAB PERBANDINGAN ===
    with tab_perbandingan:
        st.subheader("🔄 Perbandingan Data Perdagangan")

        # --- PILIHAN JENIS PERBANDINGAN ---
        comparison_type = st.selectbox(
            "Pilih Jenis Perbandingan:",
            options=["Ekspor antar Tahun", "Impor antar Tahun"],
            key="compare_type_select"
        )
        st.markdown("---") # Pemisah

        # Hanya jalankan jika data relevan ada
        if (data_ekspor and comparison_type == "Ekspor antar Tahun") or \
           (data_impor and comparison_type == "Impor antar Tahun"):

            # --- KONTEN BERDASARKAN PILIHAN ---

            # === 1. Ekspor antar Tahun ===
            if comparison_type == "Ekspor antar Tahun":
                st.markdown("##### Bandingkan Tren Total Ekspor antar Tahun")
                df_e_comp = data_ekspor["df"].copy()
                try:
                    years_available_e = sorted(df_e_comp['Tahun'].unique())
                    if not years_available_e:
                        st.warning("Tidak ada data tahun ekspor tersedia.")
                    else:
                        tahun_comp_e = st.multiselect(
                            "Pilih Tahun Ekspor untuk Dibandingkan:",
                            options=years_available_e,
                            default=[], # Default kosong
                            key="comp_year_e_multi"
                        )

                        if tahun_comp_e:
                            df_filtered_e = df_e_comp[df_e_comp['Tahun'].isin(tahun_comp_e)].copy()

                            # Buat pivot table untuk plot: Bulan sebagai index, Tahun sebagai kolom
                            # Urutkan bulan
                            bulan_order = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                                           'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
                            df_filtered_e['Bulan'] = pd.Categorical(df_filtered_e['Bulan'], categories=bulan_order, ordered=True)
                            df_pivot_e = df_filtered_e.pivot_table(index='Bulan', columns='Tahun', values='Total', aggfunc='mean')

                            if df_pivot_e.empty:
                                st.warning("Tidak ada data ekspor yang cocok untuk tahun terpilih.")
                            else:
                                fig_e_multi = go.Figure()
                                for year in df_pivot_e.columns:
                                    fig_e_multi.add_trace(go.Scatter(x=df_pivot_e.index, y=df_pivot_e[year], mode='lines+markers', name=str(year)))

                                fig_e_multi.update_layout(
                                    title="Perbandingan Rata-Rata Total Ekspor Bulanan antar Tahun",
                                    xaxis_title="Bulan",
                                    yaxis_title="Rata-Rata Total Ekspor (Juta USD)",
                                    height=450,
                                    margin=dict(t=50, b=10)
                                )
                                st.plotly_chart(fig_e_multi, use_container_width=True)

                                st.markdown("###### Tabel Rata-Rata Bulanan (Juta USD)")
                                st.dataframe(df_pivot_e.round(2))

                                st.divider(); st.subheader("💾 Download Data Perbandingan Ekspor antar Tahun")
                                csv_e_multi = df_pivot_e.reset_index().to_csv(index=False).encode('utf-8')
                                st.download_button("📥 Download (.csv)", csv_e_multi, f"perbandingan_ekspor_tahunan_{pd.Timestamp.now():%Y%m%d_%H%M}.csv", "text/csv", key="dl_e_multi")
                        else:
                             st.info("☝️ Silakan pilih tahun ekspor untuk dibandingkan.")
                except Exception as e:
                    st.error(f"Error memproses perbandingan Ekspor antar Tahun: {e}")

            # === 2. Impor antar Tahun ===
            elif comparison_type == "Impor antar Tahun":
                st.markdown("##### Bandingkan Tren Total Impor antar Tahun")
                df_i_comp = data_impor["df"].copy()
                try:
                    years_available_i = sorted(df_i_comp['Tahun'].unique())
                    if not years_available_i:
                        st.warning("Tidak ada data tahun impor tersedia.")
                    else:
                        tahun_comp_i = st.multiselect(
                            "Pilih Tahun Impor untuk Dibandingkan:",
                            options=years_available_i,
                            default=[], # Default kosong
                            key="comp_year_i_multi"
                        )

                        if tahun_comp_i:
                            df_filtered_i = df_i_comp[df_i_comp['Tahun'].isin(tahun_comp_i)].copy()

                            bulan_order = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                                           'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
                            df_filtered_i['Bulan'] = pd.Categorical(df_filtered_i['Bulan'], categories=bulan_order, ordered=True)
                            df_pivot_i = df_filtered_i.pivot_table(index='Bulan', columns='Tahun', values='Total', aggfunc='mean')

                            if df_pivot_i.empty:
                                st.warning("Tidak ada data impor yang cocok untuk tahun terpilih.")
                            else:
                                fig_i_multi = go.Figure()
                                for year in df_pivot_i.columns:
                                    fig_i_multi.add_trace(go.Scatter(x=df_pivot_i.index, y=df_pivot_i[year], mode='lines+markers', name=str(year)))

                                fig_i_multi.update_layout(
                                    title="Perbandingan Rata-Rata Total Impor Bulanan antar Tahun",
                                    xaxis_title="Bulan",
                                    yaxis_title="Rata-Rata Total Impor (Juta USD)",
                                    height=450,
                                    margin=dict(t=50, b=10)
                                )
                                st.plotly_chart(fig_i_multi, use_container_width=True)

                                st.markdown("###### Tabel Rata-Rata Bulanan (Juta USD)")
                                st.dataframe(df_pivot_i.round(2))

                                st.divider(); st.subheader("💾 Download Data Perbandingan Impor antar Tahun")
                                csv_i_multi = df_pivot_i.reset_index().to_csv(index=False).encode('utf-8')
                                st.download_button("📥 Download (.csv)", csv_i_multi, f"perbandingan_impor_tahunan_{pd.Timestamp.now():%Y%m%d_%H%M}.csv", "text/csv", key="dl_i_multi")
                        else:
                             st.info("☝️ Silakan pilih tahun impor untuk dibandingkan.")
                except Exception as e:
                    st.error(f"Error memproses perbandingan Impor antar Tahun: {e}")

        # Kondisi jika data awal tidak ada
        else:
            if comparison_type == "Ekspor antar Tahun":
                 st.warning("Data Ekspor harus berhasil dimuat untuk menampilkan perbandingan ini.")
            elif comparison_type == "Impor antar Tahun":
                 st.warning("Data Impor harus berhasil dimuat untuk menampilkan perbandingan ini.")

# --- Halaman Error Awal ---
else:
    st.error("❌ Aplikasi gagal memuat data ekspor dan impor.")
    st.markdown(f"Pastikan file `{EKSPOR_FILE_PATH}` dan `{IMPOR_FILE_PATH}` ada di direktori yang sama dengan `app.py`.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1.5rem 0;'>
    <p><strong>Analisis Pola Ekspor-Impor Indonesia | K-Means Clustering</strong></p>
</div>
""", unsafe_allow_html=True)