# Flask API Project

This project is a Flask-based API for evaluating IELTS essays, correcting grammar, and generating model answers. It utilizes various libraries for natural language processing and machine learning.

## Project Structure

```
flask-api-project
├── app
│   ├── __init__.py
│   ├── api.py
│   └── models.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd flask-api-project
   ```

2. **Install dependencies:**
   You can install the required dependencies using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application:**
   You can run the application locally using:
   ```
   python -m flask run
   ```

## Docker Deployment

To deploy the application using Docker, follow these steps:

1. **Build the Docker image:**
   ```
   docker build -t flask-api-project .
   ```

2. **Run the Docker container:**
   ```
   docker run -p 5000:5000 flask-api-project
   ```

3. **Using Docker Compose:**
   You can also use Docker Compose to run the application:
   ```
   docker-compose up
   ```

## Usage

Once the application is running, you can access the API endpoints at `http://localhost:5000`. The available endpoints include:

- **Evaluate IELTS Essay:** `/evaluate`
- **Correct Grammar:** `/correct`
- **Generate Model Answer:** `/generate`

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.