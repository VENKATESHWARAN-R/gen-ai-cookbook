import os

# DB configs
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "myappdb")
DB_USER = os.getenv("DB_USER", "myuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mypass")


DB_CONFIG = {
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "host": DB_HOST,
    "port": DB_PORT,
}


# LLm Config
BACKEND_API_BASE_URL = os.getenv(
    "BACKEND_API_BASE_URL", "http://localhost:8000"
)  # Backend API base URL
BACKEND_API_TIMEOUT = int(
    os.getenv("BACKEND_API_TIMEOUT", "60")
)  # Backend API timeout in seconds
BACKEND_API_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}  # Headers for API requests