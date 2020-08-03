from general_functions import *

if __name__ == '__main__':
    # hypothetical_goal = [0 if _ < 80 else 1 for _ in range(120)]

    MODEL_NAME = 'model/tf_idf_1.csv'
    N_ARRANGE = (1, 1)
    MODE = 'word'

    make_tf_idf_model(N_ARRANGE, MODEL_NAME, mode=MODE)

    data = pd.read_csv(MODEL_NAME, index_col=0)

    from sklearn.cluster import KMeans
    from sklearn.cluster import SpectralClustering
    from sklearn.decomposition import PCA
    from sklearn.cluster import Birch

    pca = PCA(n_components=3)
    data = pd.DataFrame(pca.fit_transform(data))
    spectral = SpectralClustering(2, random_state=0)
    k_means = KMeans(n_clusters=2, random_state=0)
    birch = Birch(threshold=0.1, n_clusters=2)

    train_and_show(data, spectral)
    train_and_show(data, k_means)
    train_and_show(data, birch)
