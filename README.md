V-Try: AI-Powered Virtual Try-OnA sophisticated web application that allows users to virtually try on clothing from any source using a photo of themselves. This project leverages a powerful Python backend for AI-driven image processing and a sleek, modern React frontend for a seamless user experience.(This image is a representation of the final UI design)
✨ FeaturesRealistic Head Compositing: Swaps the user's entire head onto a model's body for a photorealistic result.Universal Clothing Sources: Supports default models, user-uploaded photos, and links from shopping websites.Advanced AI Pipeline: Employs a multi-stage AI process for highly accurate and seamless image generation.Modern Single-Page UI: A fast, responsive interface built with React and styled with Tailwind CSS in a "Neon Tech" theme.
🤖 How It WorksThe application's realism is achieved through a precise AI pipeline. It first creates a "headless" version of the clothing model by digitally removing the head using MediaPipe for pose and segmentation analysis. Simultaneously, it processes the user's photo, straightening the head and creating a perfect background-free cutout with rembg. Finally, OpenCV calculates the exact alignment and composites the user's head onto the model's body for a seamless virtual try-on.🛠️ Tech StackBackend: Python, Flask, OpenCV, MediaPipe, Rembg, ONNX RuntimeFrontend: React, Tailwind CSS, Babel
🚀 Getting StartedTo get a local copy up and running, follow these simple steps.PrerequisitesPython 3.9+pip and venvInstallation & SetupClone the repository:git clone [https://github.com/your-username/v-try-project.git](https://github.com/your-username/v-try-project.git)
cd v-try-project
Create and activate a Python virtual environment:# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install the required Python libraries:pip install -r requirements.txt
Download the Default Model Images:Download female_model.jpg and male_model.jpg from the links provided in the project setup.
Place them in the root directory alongside app.py.Run the Flask Backend Server:python app.py
The server will start on http://127.0.0.1:5000. 
Keep this terminal running.Run the Frontend:Simply open the index.html file in your web browser.You are now ready to start using the V-Try application!
