from database import init_db

if __name__ == "__main__":
    print("Creating database and tables...")
    init_db()
    print("Done! You should now see 'blog_factory.db' in your folder.")