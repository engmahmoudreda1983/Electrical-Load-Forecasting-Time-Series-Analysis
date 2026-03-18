GLOBAL_GRID_CONFIG = {
    "Africa": {
        "Egypt": {"lat": 30.04, "lon": 31.23, "base": 25000, "cool_k": 1200, "heat_k": 50, "growth": 0.035}, 
        "South Africa": {"lat": -26.20, "lon": 28.04, "base": 30000, "cool_k": 200, "heat_k": 800, "growth": 0.015}, # Winter-peaking (Southern Hemisphere)
        "Morocco": {"lat": 33.57, "lon": -7.58, "base": 15000, "cool_k": 350, "heat_k": 150, "growth": 0.04},
        "Nigeria": {"lat": 9.08, "lon": 8.67, "base": 12000, "cool_k": 400, "heat_k": 10, "growth": 0.05},
        "Kenya": {"lat": -1.29, "lon": 36.82, "base": 8000, "cool_k": 150, "heat_k": 50, "growth": 0.045}
    },
    "Asia": {
        "Saudi Arabia": {"lat": 24.71, "lon": 46.67, "base": 45000, "cool_k": 1800, "heat_k": 20, "growth": 0.04},
        "UAE": {"lat": 25.20, "lon": 55.27, "base": 20000, "cool_k": 1500, "heat_k": 10, "growth": 0.03},
        "India": {"lat": 28.61, "lon": 77.20, "base": 160000, "cool_k": 2500, "heat_k": 100, "growth": 0.06},
        "Japan": {"lat": 35.67, "lon": 139.65, "base": 90000, "cool_k": 800, "heat_k": 1500, "growth": 0.002},
        "China": {"lat": 39.90, "lon": 116.40, "base": 500000, "cool_k": 4000, "heat_k": 5000, "growth": 0.05}
    },
    "Europe": {
        "Germany": {"lat": 52.52, "lon": 13.40, "base": 55000, "cool_k": 50, "heat_k": 3500, "growth": 0.005}, # Heavy Winter Peak
        "France": {"lat": 48.85, "lon": 2.35, "base": 50000, "cool_k": 80, "heat_k": 4000, "growth": 0.006},  # Heavy Winter Peak
        "UK": {"lat": 51.50, "lon": -0.12, "base": 35000, "cool_k": 30, "heat_k": 2800, "growth": 0.005},
        "Italy": {"lat": 41.90, "lon": 12.49, "base": 40000, "cool_k": 800, "heat_k": 1500, "growth": 0.004},
        "Spain": {"lat": 40.41, "lon": -3.70, "base": 30000, "cool_k": 1000, "heat_k": 800, "growth": 0.008}
    },
    "North America": {
        "USA": {"lat": 39.00, "lon": -100.00, "base": 450000, "cool_k": 6000, "heat_k": 3000, "growth": 0.01}, # Summer Peaking
        "Canada": {"lat": 43.65, "lon": -79.38, "base": 70000, "cool_k": 100, "heat_k": 5000, "growth": 0.012}, # Heavy Winter Peak
        "Mexico": {"lat": 23.63, "lon": -102.55, "base": 40000, "cool_k": 900, "heat_k": 100, "growth": 0.025}
    }
}