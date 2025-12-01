from ndw.incidents import load_incidents, load_active_incidents

# All incidents
df_all = load_incidents()
print(df_all.head())
print(len(df_all), "total incidents")

# Only active incidents with coordinates
df_active = load_active_incidents()
print(df_active[["id", "creation_time", "lat", "lon", "carriageway", "direction_ref"]].head())