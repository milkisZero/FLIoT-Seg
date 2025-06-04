   pyinstaller --onefile --add-data "pickle_utils.py;." --add-data "datasetCICIDS.py;." --add-data "requirements.txt;." \
   --hidden-import=tensorflow --hidden-import=tensorflow_core --hidden-import=tensorflow_core.keras \
   --hidden-import=tensorflow_core.keras.layers --hidden-import=tensorflow_core.keras.models \
   --hidden-import=tensorflow_core.keras.utils --hidden-import=keras --hidden-import=numpy \
   --hidden-import=pandas --hidden-import=matplotlib --hidden-import=psutil fl_client.py