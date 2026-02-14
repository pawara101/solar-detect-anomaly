## Solar PV EL Anomaly Detection App

This application is designed to detect anomalies in Electroluminescence (EL) images of solar photovoltaic (PV) modules using a YOLO-based object detection model. The app provides an interactive interface for users to upload EL images, view detected anomalies, and analyze the results.

### Features
- **Image Upload**: Users can upload EL images of solar PV modules for analysis.
- **Anomaly Detection**: The app uses a pre-trained YOLO model to identify and classify anomalies in the uploaded images.
- **Visualization**: Detected anomalies are highlighted in the images, allowing users to easily identify issues.
- **User-Friendly Interface**: The app is built with Gradio, providing an intuitive and responsive user interface.
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/solar-anomaly-detect-app.git
    cd solar-anomaly-detect-app
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the application:
    ```bash
    python app.py
    ```
### Dockerization
To run the application using Docker, follow these steps:
1. Build the Docker image:
   ```bash
   docker build -t solar-anomaly-detect-app .
   ```
2. Run the Docker container:
   ```bash
    docker run -p 7860:7860 solar-anomaly-detect-app
    ```
### Usage
1. Open a web browser and navigate to `http://localhost:7860`.
2. Upload an EL image of a solar PV module.
3. View the detected anomalies highlighted in the image.
4. Analyze the results and take necessary actions based on the detected anomalies.