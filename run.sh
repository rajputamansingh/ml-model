#!/bin/bash

# Script to set up and run the Brain Disorder ML Analysis app

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up Brain Disorder ML Analysis App...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Check if virtual environment exists, if not create it
if [ ! -d "venv" ]
then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Please check your Python installation.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment.${NC}"
    exit 1
fi

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install dependencies.${NC}"
    exit 1
fi
echo -e "${GREEN}Dependencies installed successfully.${NC}"

# Run setup script to create sample images
echo -e "${YELLOW}Running setup script...${NC}"
python setup.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to run setup script.${NC}"
    exit 1
fi
echo -e "${GREEN}Setup completed successfully.${NC}"

# Create .streamlit directory if it doesn't exist
if [ ! -d ".streamlit" ]; then
    echo -e "${YELLOW}Creating .streamlit directory...${NC}"
    mkdir -p .streamlit
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create .streamlit directory.${NC}"
        exit 1
    fi
fi

# Check if config.toml exists
if [ ! -f ".streamlit/config.toml" ]; then
    echo -e "${YELLOW}Config file not found. Creating default config...${NC}"
    # Just check if the directory exists, create it if not
    if [ ! -d ".streamlit" ]; then
        mkdir -p .streamlit
    fi
    cp config.toml .streamlit/config.toml 2>/dev/null || echo "Warning: config.toml template not found"
fi

# Run the app
echo -e "${GREEN}Starting the Brain Disorder ML Analysis App...${NC}"
streamlit run app.py

# Deactivate virtual environment when done
deactivate
