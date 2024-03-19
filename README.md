# Celonis Gesture Recognition ML Challenge

## Overview
"Celnois Gesture Recognition" is a FastAPI-based application designed to tackle assignment. It is containerized with Docker for easy deployment and scaling.

## Getting Started

### Prerequisites
- Docker installed on your machine.

### Installation and Running

1. **Build the Docker image:**
   ```bash
   docker build -t myfastapiapp .
   ```

2. **Run the container:**
   ```bash
   docker run -d --name myfastapiapp -p 80:80 myfastapiapp
   ```

Your application is now running and accessible at `http://localhost/docs`.
