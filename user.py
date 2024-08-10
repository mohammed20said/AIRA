import mysql.connector
from configparser import ConfigParser

# Initialize CNX as None
CNX = None

def connect_to_db():
    global CNX
    if CNX is None:
        config = ConfigParser()
        config.read("config.ini")
        _host = config.get('MySQL', 'host')
        _port = config.get('MySQL', 'port')
        _database = config.get('MySQL', 'database')
        _user = config.get('MySQL', 'user')
        _password = config.get('MySQL', 'password')
        CNX = mysql.connector.connect(
            host=_host,
            database=_database,
            user=_user,
            passwd=_password,
            port=_port
        )
        
def update_password(username, old_password, new_password):
    try:
        connect_to_db()
        with CNX.cursor() as cur:
            cur.execute(
                "UPDATE Users SET Password = %s WHERE UserId = %s AND Password = %s",
                (new_password, username, old_password)
            )
            CNX.commit()
            return cur.rowcount > 0  # Returns True if the update was successful, False otherwise
    except Exception as e:
        print(f"An error occurred during sign-up: {e}")
        return False
def sign_up(userName: str, password: str) -> bool:
    try:
        connect_to_db()
        with CNX.cursor() as cur:
            # Check if the user already exists
            cur.execute("SELECT COUNT(*) FROM Users WHERE UserId = %s", (userName,))
            if cur.fetchone()[0] > 0:
                # User exists, update password
                cur.execute(
                    "UPDATE Users SET Password = %s, CreatedTime = NOW(), Active = 1 WHERE UserId = %s",
                    (password, userName)
                )
            else:
                # User does not exist, insert new user
                cur.execute(
                    "INSERT INTO Users (UserId, Password, CreatedTime, Active) VALUES (%s, %s, NOW(), 1)",
                    (userName, password)
                )
            CNX.commit()
            return True
    except Exception as e:
        print(f"An error occurred during sign-up: {e}")
        return False

def login(userName: str, password: str) -> bool:
    try:
        connect_to_db()
        with CNX.cursor() as cur:
            # Check user credentials
            cur.execute(
                "SELECT COUNT(*) FROM Users WHERE UserId = %s AND Password = %s AND Active = 1",
                (userName, password)
            )
            return cur.fetchone()[0] > 0
    except Exception as e:
        print(f"An error occurred during login: {e}")
        return False