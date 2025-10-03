from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Use the standard sqlite driver (no aiosqlite) for simple sync access:
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

# create_engine will use the builtin sqlite3 module
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed only for SQLite + multiple threads
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# example table
class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)

def init_db():
    Base.metadata.create_all(bind=engine)
