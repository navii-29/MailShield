from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import predictpipeline, custom_data

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['GET','POST'])
def predict_email():
    try:
        # Get input from form
        email_text = request.form.get("email")

        if not email_text:
            return render_template(
                "home.html",
                error="No email text provided"
            )

        # Prepare data
        data = custom_data(email_text)
        input_email = data.get_data_as_string()

        # Predict
        predictor = predictpipeline()
        result, prob = predictor.initiate_predict_pipeline(input_email)

        # Render result
        return render_template(
            "home.html",
            results=result,
            raw_prediction=prob
        )

    except Exception as e:
        return render_template(
            "home.html",
            error=str(e)
        )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5011, debug=True)
