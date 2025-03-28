import sys
import random


def read_input_file(filename):
    """입력 파일에서 벡터를 읽어와 리스트로 반환합니다."""
    vectors = []
    with open(filename, "r") as file:
        for line in file:
            vectors.append([int(x) for x in line.split()])
    return vectors


def initialize_centroids(vectors, k):
    """
    초기 클러스터 중심 (centroids)을 랜덤으로 선택합니다.
    """
    return random.sample(vectors, k)


def calculate_distance(vector1, vector2):
    """
    두 벡터 간의 유클리드 거리(Euclidean distance)를 계산합니다.
    """
    return sum((x - y) ** 2 for x, y in zip(vector1, vector2)) ** 0.5


def assign_clusters(vectors, centroids):
    """
    각 벡터를 가장 가까운 클러스터 중심에 할당합니다.
    """
    clusters = [[] for _ in range(len(centroids))]
    for vector in vectors:
        distances = [calculate_distance(vector, centroid) for centroid in centroids]
        closest_index = distances.index(min(distances))
        clusters[closest_index].append(vector)
    return clusters


def update_centroids(clusters):
    """
    각 클러스터의 중심 좌표를 업데이트합니다.
    """
    new_centroids = []
    for cluster in clusters:
        if cluster:  # 클러스터가 비어있지 않은 경우
            centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
            new_centroids.append(centroid)
        else:  # 비어있는 클러스터는 그대로 둠
            new_centroids.append([0] * 5)
    return new_centroids


def kmeans_clustering(filename, k, max_iterations):
    """
    K-means 클러스터링 알고리즘의 메인 함수입니다.
    """
    vectors = read_input_file(filename)
    centroids = initialize_centroids(vectors, k)

    for iteration in range(max_iterations):
        clusters = assign_clusters(vectors, centroids)
        updated_centroids = update_centroids(clusters)

        # 중심이 변하지 않으면 조기 종료
        if updated_centroids == centroids:
            print(f"# of actual iterations: {iteration + 1}")
            break
        centroids = updated_centroids
    else:
        print(f"# of actual iterations: {max_iterations} (max reached)")

    # 결과 출력
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: representative = {centroids[i]}, # of vectors = {len(cluster)}")


if __name__ == "__main__":
    # 커맨드라인 인자로부터 입력을 받습니다.
    if len(sys.argv) != 4:
        print("Usage: python kmeansClustering.py inputfile.txt [value of k] [# of iterations]")
        sys.exit(1)

    input_file = sys.argv[1]
    k_value = int(sys.argv[2])
    max_iters = int(sys.argv[3])

    kmeans_clustering(input_file, k_value, max_iters)
