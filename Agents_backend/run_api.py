import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "Api:app",
        # host="0.0.0.0",
        port=8000,
        reload=True
    )
