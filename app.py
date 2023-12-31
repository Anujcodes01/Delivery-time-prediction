from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
import os,sys
from src.config.configuration import *
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from flask import Flask,render_template,request
from src.pipeline.prediction_pipeline import CustomData,PredictionPipeline
from werkzeug.utils import secure_filename
from Batch_Prediction.batch import *
from src.pipeline.training_pipline import Train

feature_engineering_file_path = FEATURE_ENG_OBJ_FILE_PATH
transformer_file_path= PREPROCESSING_OBJ_FILE
model_file_path = MODEL_FILE_PATH

UPLOAD_FOLDER = "batch_prediction/UPLODED_CSV_FILE"

app = Flask(__name__,template_folder='templates')

ALLOWED_EXTENSION = {'csv'}

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])


def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        data = CustomData(
            Delivery_person_Age = int(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings =float(request.form.get('Delivery_person_Ratings')),
            Weather_conditions=int(request.form.get('Weather_conditions')),
            Vehicle_condition=request.form.get('Vehicle_condition'),
            multiple_deliveries=int(request.form.get('multiple_deliveries')),
            Road_traffic_density=request.form.get('Road_traffic_density'),
            distance=float(request.form.get('distance')),
            Type_of_order=request.form.get('Type_of_order'),
            Type_of_vehicle=request.form.get('Type_of_vehicle'),
            Festival=request.form.get('Festival'),
            City=request.form.get('City')
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)

        result = int(pred[0])

        return render_template('form.html',finel_result = result)
    
@app.route("/batch",methods=['GET','POST'])
def perform_batch_prediction():

    if request.method == 'GET':
        return render_template('batch.html')
    else:
        file = request.files['csv_file']
        directory_path = UPLOAD_FOLDER
        os.makedirs(directory_path,exist_ok=True)

        if file and '.' in file.filename and file.filename.replit('.',1)[1].lower() in ALLOWED_EXTENSION:
            for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
                file_path = os.path.join(UPLOAD_FOLDER,filename)
                if file.save(file_path):
                    os.remove(file_path)

            filename=secure_filename(file.filename)
            file_path=os.path.join(UPLOAD_FOLDER,filename)
            file.save(file_path)
            print(file_path)

            logging.info("csv recieved and uploaded")

            batch= batch_prediction(file_path,
                                   model_file_path,
                                   transformer_file_path,
                                   feature_engineering_file_path)
            batch.start_batch_prediction()

            output = "batch Prediction Done"
            return render_template("batch,html",prediction_result = output,prediction_type='batch')
        else:
            return render_template("batch.html",predictiton_type='batch',error='Invalid File type')
        

@app.route("/train",methods=['GET','POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html')
    else:
        try:
            pipeline = Train()
            pipeline.main()

            return render_template('train.html',message='training complete')
        except Exception as e:
            logging.error(f"{e}")
            error_message = str(e)
            return render_template('index.html',error=error_message)
        
if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True,port='8888')
