import datetime
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database location
SQLALCHEMY_DATABASE_URL = "sqlite:///./blog_factory.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class BlogRecord(Base):
    __tablename__ = "blogs"

    id = Column(String, primary_key=True, index=True)
    topic = Column(String, index=True)
    content = Column(Text, nullable=True)      # The blog text
    score = Column(Float, nullable=True)        # The 9.4/10
    verdict = Column(String, nullable=True)     # "✅ PASSED"
    fact_check = Column(Text, nullable=True)    # The NLI report
    status = Column(String, default="PENDING") # PENDING, COMPLETED, FAILED
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)