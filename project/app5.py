import numpy as np
from flask import Flask, request, render_template, session
import pickle
import google.generativeai as genai
from markupsafe import Markup

API_KEY = 'AIzaSyDGGjEPCKkGjnM8jvM7jgqj-wPW_gJ3OGs'

app = Flask(__name__, template_folder="templates")
app.secret_key = "your_secret_key"  # Required for session handling

model = pickle.load(open('weights2.pkl', 'rb'))


@app.route('/')
def h():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/index')
def home():
    return render_template('index2.html')


@app.route('/predict', methods=['GET'])
def predict():
    logical_quotient_rating = int(request.args.get('Logical quotient Skills', 0))
    coding_skills_rating = int(request.args.get('Coding Skills', 0))
    public_speaking_points = int(request.args.get('Public Speaking', 0))
    self_learning_capability = int(request.args.get('Self Learning Capability', 0))
    extra_courses = int(bool(request.args.get('Extra courses', "").strip()))
    certifications = int(bool(request.args.get('certifications', "").strip()))
    reading_writing_skills = int(request.args.get('Reading and writing skills', 0))
    memory_capability_score = int(request.args.get('Memory capability score', 0))
    interested_subjects = int(bool(request.args.get('Interested Subjects', "").strip()))
    interested_career_area = int(bool(request.args.get('Interested Career Area', "").strip()))
    company_type = int(bool(request.args.get('Type of Company You Want to Settle In', "").strip()))
    taken_inputs_from_seniors = int(bool(request.args.get('Took advice from seniors or elders', "").strip()))
    interested_books = int(bool(request.args.get('Interested Books Category', "").strip()))
    worked_in_teams = int(bool(request.args.get('Team Co-ordination Skill', "").strip()))
    introvert = int(bool(request.args.get('Introvert', "").strip()))
    hackathons = int(bool(request.args.get('Hackathons participated', "").strip()))
    workshop = int(bool(request.args.get('Workshops Attended', "").strip()))

    # Input array for prediction
    arr = np.array([
        logical_quotient_rating, coding_skills_rating,
        public_speaking_points, self_learning_capability, extra_courses,
        certifications, reading_writing_skills, memory_capability_score,
        interested_subjects, interested_career_area, company_type,
        taken_inputs_from_seniors, interested_books, worked_in_teams,
        introvert, hackathons, workshop
    ]).reshape(1, -1)

    print("arr", arr)

    # Make prediction
    output = model.predict(arr)[0]  # Get single prediction
    session['predicted_job'] = output  # Store in session

    return render_template('output.html', output=f'Your job is {output}')


@app.route('/suggestion', methods=['GET'])
def suggestion():
    predicted_job = session.get('predicted_job', None)  # Retrieve from session
    if predicted_job is None:
        return "No prediction found. Please visit /predict first.", 400

    genai.configure(api_key=API_KEY)
    modelling = genai.GenerativeModel("gemini-1.5-pro")
    project_suggestions = modelling.generate_content(
        f"Suggest some project ideas for a BTech student interested in a career in {predicted_job}. Provide some project ideas capital letters of font size 1.5 rem along with description of 50 words and abstract of 100 words."
    )
    suggested_text = project_suggestions.candidates[0].content.parts[0].text
    out2 = f'Here are some Project ideas and its Abstract for an aspiring {predicted_job} <br> {suggested_text}'
    output2 = Markup(out2.replace("\n", "<br>").replace("*", "")).strip()

    return render_template('output2.html', output2=output2)


@app.route('/routes', methods=['GET'])
def show_routes():
    return str(app.url_map)


if __name__ == "__main__":
    app.run(debug=True)
