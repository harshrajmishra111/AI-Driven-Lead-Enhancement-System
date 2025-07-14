# AI-Driven-Lead-Enhancement-System

Overview
This project enhances Caprae Capital’s lead generation process by developing a tool to fetch, refine, and present high-quality leads for sale to businesses, aligning with their AI-driven post-acquisition growth strategy. I created two components: a CLI-based lead enhancer (lead_enhancer.py) and a Streamlit web app (app.py). These tools improve lead scoring, clustering, and visualization with dynamic UI graphs and diverse tables, helping clients explore leads easily. Additionally, I’ve worked on a personal churn detection project, briefly integrated to suggest retention strategies for subscription businesses, potentially unlocking new lead data for Caprae’s expansion.

Features
Lead Scoring: Classifies leads into actionable categories using a simple, effective model.
Clustering: Segments leads into 3 groups for tailored offerings, displayed in intuitive tables.
Visualization: Streamlit app offers UI graphs (e.g., scatter plots) and tables (e.g., industry trends) for client-friendly lead exploration.
Churn Detection: Provides basic retention suggestions for subscription businesses, with potential to legally gather lead data for resale.
Data Management: CLI supports filtering, exporting, and managing leads efficiently.
Prerequisites
Before running the project, ensure you have the following installed:

Python 3.12+
pip (Python package manager)

Installation
Install Dependencies: Create a virtual environment (optional but recommended) and install required packages:

Run
pip install -r requirements.txt

Prepare the Dataset: Place your lead data file (leads_data.csv) in the project directory. Ensure it contains columns like email, job_title, company, bio, industry, revenue_estimate, contract_status, and contract_value.
Usage
CLI Lead Enhancer
Run the script to process leads and interact via command line:

Run
python lead_enhancer.py

Options include filter, view, export, report, trends, importance, evaluate, and exit.
Example: Type report to generate a lead report saved as lead_report.txt.
Streamlit Web App

Launch the interactive dashboard:

Run
streamlit run app.py


Explore leads with filters (industry, status, source, cluster, score), view graphs, and export data or reports.
Graphs and tables update dynamically based on filter selections.
Project Structure
lead_enhancer.py: Core logic for lead processing, scoring, and CLI interface.
app.py: Streamlit app for visual lead exploration and management.
leads_data.csv: Sample dataset (replace with your data).
requirements.txt: List of dependencies.

Notes
The tool assumes a pre-existing leads_data.csv. For full alignment with SaaSQuatch Leads, consider adding web scraping in future iterations.
The churn detection feature is a conceptual add-on; integrate it with additional data for full functionality.

Contact
I’m passionate about business innovation and eager to join Caprae Capital. Thank you for considering my work—I hope to hear from you soon!
Email - harshrajmishra.hrm@gmail.com

Email: [Your Email]
