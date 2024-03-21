from flask import Flask, render_template, request, jsonify
from RAG_app import main

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('/mnt/c/llm_SERC_documentation/sree_llm_chatbot/RAG_assignment/templates/index.html')


@app.route('/api/question', methods=['POST'])
def get_question():
    if request.method == 'POST':
        data = request.get_json()
        if 'Question' in data:
            question = data['Question']
            answer = main(question)
            response = {
                'Answer': answer
            }
            return jsonify(response)
        else:
            return jsonify({'error': 'Question field is missing'}), 400
    else:
        return jsonify({'error': 'Only POST requests are allowed'}), 405

if __name__ == "__main__":
    app.run(debug=True)




# from flask import Flask, render_template, request, jsonify
# from RAG_app import main  # Assuming this is where your main function is defined

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/api/question', methods=['POST'])
# def get_question():
#     if request.method == 'POST':
#         data = request.get_json()
#         if 'Question' in data:
#             question = data['Question']
#             answer = main(question)  # Assuming main function returns answer
#             response = {'Answer': answer}
#             return jsonify(response)
#         else:
#             return jsonify({'error': 'Question field is missing'}), 400
#     else:
#         return jsonify({'error': 'Only POST requests are allowed'}), 405

# if __name__ == "__main__":
#     app.run(debug=True)