from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import whisper
from google import genai
from dotenv import load_dotenv
import json  # Import json module for parsing
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS
CORS(app)

# Ensure the upload directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

load_dotenv()
# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

# Initialize whisper model
model = whisper.load_model("tiny")

API_KEY = os.getenv("API_KEY")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

client = genai.Client(api_key=API_KEY)

def analyze_transcript_with_prompt(transcript, prompt):
    try:
        input_text = f"{prompt}\n\nTranscript:\n{transcript}"

        # Generate analysis using the Gemini API
        analysis = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[input_text],
        )
        return analysis.text
    except Exception as e:
        print(f"Error analyzing transcript with Gemini: {e}")
        return None

def analyze_sales_calls(audio_file):
    try:
        # Transcribe the audio file
        result = model.transcribe(audio_file)
        transcript = result['text']
        print(transcript)

        prompt = """
        Analyze the following sales call transcript and provide metrics, feedback, and recommendations using Alex Hormozi's proven frameworks
        in the following parameters:
                1. Metrics: Provide metrics on the sales call, such as the number of questions asked, the number of objections raised, and the number of times the customer spoke.
                2. Feedback: Provide feedback on the sales call, such as the salesperson's tone, pace, and energy level.
                3. Recommendations: Provide recommendations on how the salesperson can improve, such as asking more open-ended questions, active listening, and handling objections effectively.
        Use **bold** for emphasis and *italics* for subtle emphasis.
        Return data in this format:
                {
                metrics: [
                  {
                    name: "Building Rapport",
                    score: 85,
                    feedback: "Strong opening and consistent engagement throughout the call.",
                  },
                  {
                    name: "Pain Point Identification",
                    score: 70,
                    feedback: "Good probing questions, but missed some key pain points.",
                  },
                  {
                    name: "Value Proposition",
                    score: 90,
                    feedback: "Excellent presentation of value and benefits.",
                  },
                  {
                    name: "Objection Handling",
                    score: 65,
                    feedback: "Consider using the Feel, Felt, Found framework more consistently.",
                  }
                ],
                overallScore: 78,
                recommendations: [
                  "Use more specific examples when addressing objections",
                  "Implement the *Feel*, *Felt*, *Found* framework for handling objections",
                  "Ask more probing questions to uncover deeper pain points",
                  "Practice assumptive closing techniques",
                ]
              }
              :
        """
        analysis_result = analyze_transcript_with_prompt(transcript, prompt)

        # Clean and parse the JSON response
        cleaned_result = analysis_result.strip().replace("```json", "").replace("```", "")
        parsed_result = json.loads(cleaned_result)  # Parse the JSON string into a Python dictionary

        return {
            "transcript": transcript,
            "analysis": parsed_result  # Return the parsed JSON data
        }

    except Exception as e:
        print(f"Error analyzing sales calls: {e}")
        return None


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Convert relative path to absolute path
        abs_filepath = os.path.abspath(filepath)
        abs_filepath = abs_filepath.replace("\\", "/")

        analysis_result = analyze_sales_calls(abs_filepath)

        if analysis_result:
            return jsonify(analysis_result), 200
        else:
            return jsonify({"error": "Error analyzing audio file"}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)