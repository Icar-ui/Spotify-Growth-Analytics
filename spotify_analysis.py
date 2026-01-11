import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. صاوب بيانات تجريبية (حيت مازال ماعندكش ملف)
data = {
    'user_id': range(1, 101),
    'hours_listened_weekly': [10, 2, 15, 1, 20, 5, 12, 3, 25, 30] * 10,
    'songs_skipped_per_day': [2, 10, 1, 15, 0, 12, 2, 18, 1, 0] * 10
}

df = pd.DataFrame(data)

# 2. تطبيق الـ KMeans (تقسيم المستخدمين لمجموعات)
# غانقسموهم لـ 3 مجموعات: (Fans, Normal, At Risk)
kmeans = KMeans(n_clusters=3, random_state=42)
df['segment'] = kmeans.fit_predict(df[['hours_listened_weekly', 'songs_skipped_per_day']])

# 3. عرض النتائج
print("First 5 rows of our analysis:")
print(df.head())

# 4. رسم مبياني بسيط
plt.scatter(df['hours_listened_weekly'], df['songs_skipped_per_day'], c=df['segment'])
plt.xlabel('Hours Listened Weekly')
plt.ylabel('Songs Skipped')
plt.title('Spotify User Segmentation for Marketing')
plt.show()