@echo off
ECHO Setting up Brain Disorder ML Analysis App...

REM Check if Python is installed
WHERE python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    ECHO Python is not installed or not in PATH. Please install Python and try again.
    EXIT /B 1
)

REM Check if virtual environment exists, if not create it
IF NOT EXIST venv (
    ECHO Creating virtual environment...
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        ECHO Failed to create virtual environment. Please check your Python installation.
        EXIT /B 1
    )
    ECHO Virtual environment created.
)

REM Activate virtual environment
ECHO Activating virtual environment...
CALL venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    ECHO Failed to activate virtual environment.
    EXIT /B 1
)

REM Install dependencies
ECHO Installing dependencies...
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    ECHO Failed to install dependencies.
    EXIT /B 1
)
ECHO Dependencies installed successfully.

REM Run setup script to create sample images
ECHO Running setup script...
python setup.py
IF %ERRORLEVEL% NEQ 0 (
    ECHO Failed to run setup script.
    EXIT /B 1
)
ECHO Setup completed successfully.

REM Create .streamlit directory if it doesn't exist
IF NOT EXIST .streamlit (
    ECHO Creating .streamlit directory...
    mkdir .streamlit
    IF %ERRORLEVEL% NEQ 0 (
        ECHO Failed to create .streamlit directory.
        EXIT /B 1
    )
)

REM Check if config.toml exists
IF NOT EXIST .streamlit\config.toml (
    ECHO Config file not found. Creating default config...
    IF NOT EXIST .streamlit (
        mkdir .streamlit
    )
    COPY config.toml .streamlit\config.toml 2>nul || ECHO Warning: config.toml template not found
)

REM Run the app
ECHO Starting the Brain Disorder ML Analysis App...
streamlit run app.py

REM Deactivate virtual environment when done (will only execute if the app is closed)
deactivate
