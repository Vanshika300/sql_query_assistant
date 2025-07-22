🔍 SQL Query Assistant

An AI-powered web application that allows users to interact with a MySQL database using natural language. This tool translates user questions into executable SQL queries, runs them, and provides clear, conversational answers, making database interaction accessible to non-technical users.
✨ Features
•	Natural Language to SQL Translation: Converts plain English questions into syntactically correct MySQL queries using Google Gemini's powerful LLM.
•	Query Execution: Seamlessly executes the generated SQL queries against a connected MySQL database.
•	Natural Language Response Generation: Provides easy-to-understand, conversational answers based on the query results.
•	Interactive User Interface: Built with Streamlit for a user-friendly and intuitive experience.
•	Database Connection Status: Displays real-time database connection details and lists available tables.
•	Example Questions & Quick Actions: Offers pre-defined queries and quick buttons to help users explore the database efficiently.
•	Error Handling: Includes robust error handling for API and database connection issues.
🚀 Technologies Used
•	Python
•	Streamlit: For building the interactive web interface.
•	Google Generative AI (Gemini 2.5 Flash): The Large Language Model (LLM) for natural language understanding and SQL generation.
•	Langchain: Framework for orchestrating LLM interactions with the database.
•	MySQL: The relational database system.
•	python-dotenv: For managing environment variables (e.g., API keys, database credentials).
•	pymysql: MySQL database connector for Python.
•	tenacity: For adding retry logic to API calls.
📸 Screenshots
Main Interface & AI Response
A user querying "show me the total revenue" and receiving a natural language response.
Additional Insights & Quick Actions
The application providing additional insights and quick action buttons after a query.
⚙️ How to Run Locally
Follow these steps to set up and run the SQL Query Assistant on your local machine.
1. Prerequisites
•	Python 3.8+: Ensure you have Python installed.
•	MySQL Database: Have a MySQL database instance running (e.g., via Docker, XAMPP, or a local installation).
o	Make sure you have a database named mystore (or adjust the DB_NAME in your .env file).
o	Ensure your MySQL user (root by default in the code) has the necessary permissions.
•	Google Gemini API Key: Obtain an API key from Google AI Studio.
2. Clone the Repository
git clone https://github.com/your-username/sql-query-assistant.git
cd sql-query-assistant


3. Set Up Environment Variables
Create a file named .env in the root directory of your project (the same directory as sql_query.py). Add the following content, replacing the placeholder values with your actual credentials and API key:
GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
DB_USER="root"
DB_PASSWORD="your_mysql_password"
DB_HOST="localhost"
DB_PORT="3306"
DB_NAME="mystore"


Important: Never share your GOOGLE_API_KEY or database credentials publicly. The .env file is included in the .gitignore to prevent accidental uploads.
4. Install Dependencies
It's recommended to use a virtual environment to manage dependencies.
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt


If you don't have a requirements.txt file, you can generate one by running pip freeze > requirements.txt after installing the necessary libraries (streamlit, langchain-google-genai, langchain-community, SQLAlchemy, pymysql, python-dotenv, tenacity).
5. Run the Application
Once all dependencies are installed and your .env file is configured, run the Streamlit application from your terminal:
streamlit run sql_query.py


This command will open the application in your web browser, usually at http://localhost:8501.
📧 Contact
Feel free to reach out if you have any questions or feedback!
•	Email: vanshikashukla065@gmail.com
•	LinkedIn: linkedin.com/in/vanshika-shukla30
