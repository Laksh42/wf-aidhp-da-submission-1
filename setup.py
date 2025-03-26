from setuptools import setup, find_packages

setup(
    name="financial-advisor-chatbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pymongo",
        "motor",
        "python-dotenv",
        "pydantic",
        "pydantic-settings",
        "openai",
        "pandas",
        "numpy",
        "python-jose",
        "passlib",
        "python-multipart"
    ],
    python_requires=">=3.8",
    author="Financial Advisor Team",
    author_email="example@example.com",
    description="Multi-Modal Financial Advisor Chatbot",
) 