BMI Project README
📝 Introduction
This project is a BMI Calculator powered by AI, providing users with personalized health insights based on their Body Mass Index (BMI). The application allows users to calculate their BMI, view health recommendations based on their measurements, and interact with an AI chatbot to get health-related advice.

🌟 Features
Client-Side Features:
1.BMI Calculator Section

Users can enter their height and weight to calculate their BMI.

Based on the BMI value, users will be categorized into one of the following four categories:

Underweight

Normal

Overweight

Obese

2.AI Health Guidance

Users can receive AI-driven health recommendations by submitting their height, weight, age, and gender.

After filling in these details, users can click the Get AI Recommendation button to receive personalized health information.

3.BMI About Section

The About section provides users with detailed information about BMI, including what it is, how it's calculated, and its importance for health.

Backend-Side Features:
AI Health Recommendation

When a user requests health recommendations, the backend sends the user's height, weight, age, and gender to the AI system.

The AI processes the data and returns a response, which is then sent back to the client-side for display.

Chatbot AI Feature

Users can interact with a chatbot AI to ask health-related questions.

The chatbot provides users with health-related information and answers any health queries the user may have.

Tech Stack:
Frontend: Vue.js

Backend: Laravel

AI: Google Gemini API (for health recommendations)

⚙️ Installation
📥 Prerequisites:
vue.js (for frontend)

Laravel (for backend)


💻 Steps to Install:
bash
Copy
# Clone the frontend repository
git clone https://github.com/kamalkant24/BMI-AI-calculator-frontend.git

# Navigate to the project directory
cd repo-name

# Install backend dependencies
composer install

# Install frontend dependencies
npm install
# Run the frontend development server
npm run dev



# Clone the frontend repository
git clone https://github.com/kamalkant24/BMI-AI-calculator.git

# Navigate to the project directory
cd repo-name

# Set up environment variables
cp .env.example .env

# Generate the Laravel application key
php artisan key:generate

# Run migrations to set up the database
php artisan migrate

# Start the backend server
php artisan serve

