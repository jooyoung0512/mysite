import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
nba_data = pd.read_csv('Seasons_Stats.csv')

# 2. 필요한 컬럼만 선택
selected_columns = [
    'Player', 'Year', 'Pos', 'PTS', 'AST', 'TRB', 'G', 'MP', 'PER', 
    'FG%', 'FT%', 'Age', 'Tm'
]
nba_data_selected = nba_data[selected_columns]

# 3. 결측치 처리 (결측치가 있는 행 제거)
nba_data_cleaned = nba_data_selected.dropna()

# 4. 데이터 정규화 (스탯 간 스케일 차이를 보정)
scaler = StandardScaler()
nba_data_scaled = scaler.fit_transform(nba_data_cleaned.drop(columns=['Player', 'Year', 'Pos', 'Tm']))

# 5. K-NN 모델 생성 (유사한 5명의 선수 찾기)
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(nba_data_scaled)

# 6. 특정 선수를 입력 받아 유사한 선수 찾기
def find_similar_players(player_name, n=5):
    # 입력된 선수의 데이터를 추출
    player_data = nba_data_cleaned[nba_data_cleaned['Player'] == player_name].drop(columns=['Player', 'Year', 'Pos', 'Tm'])
    
    # 데이터가 없는 경우 예외 처리
    if player_data.empty:
        return f"Player {player_name} not found in the dataset."
    
    # 데이터 정규화
    scaled_player_data = scaler.transform(player_data)
    
    # 유사한 선수 찾기
    distances, indices = knn.kneighbors(scaled_player_data, n_neighbors=n)
    
    # 결과 반환 (유사한 선수들의 이름)
    similar_players = nba_data_cleaned.iloc[indices[0]]['Player'].values
    return similar_players

# 7. 선수 검색 및 추천
player_to_search = 'LeBron James'  # 원하는 선수 이름을 입력하세요
similar_players = find_similar_players(player_to_search, n=5)  # 유사한 선수 5명을 추천
print(f"'{player_to_search}'와 유사한 선수 5명: {similar_players}")