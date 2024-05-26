from flask import Flask
from flask_restx import Api, Resource, fields
#from predice_genero import PrediceGenero

# Importación librerías

#librerias de datos
import pandas as pd
import numpy as np
import sys

#librerías de procesamiento
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')

import re

from tensorflow.keras.models import load_model

#Función de calculo de generos:
def PrediceGenero(text, lematize =False, stemming = False, embedding_dim = 300):
    """
    Preprocesamiento de textos para utilizar en el modelo.

    Parametros:
    
    text(str): la cadena de texto a la que se le va a aplicar el procesamiento.
    lematize(Bool): cuando es True lematiza los verbos (default: False)
    stemming (Bool): cuando es True hace stemming a las palabras (default: False)
    embedding_dim(int): dimension de los vectores de embeddings de palabras (defalut: 300)
    """
    #ruta = 'C:/Users/SANTIAGO/MIAD/2-ML y PLN/Git/MIAD_ML_NLP_2023/Semana 7/Competencia/' #Ruta de los archivos
    embedding_dim = 300 #Dimensión de embeddings

    ruta_glove = 'glove.6B.'+str(embedding_dim)+'d.txt' #Para importar archivo de embeddings
    ruta_emb = 'NN_genre.h5' #Para importar modelo preentrenado


    embeddings_index = {} #Se guarda el diccionario de los embeddings
    with open(ruta_glove, encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    modelo_def = load_model(ruta_emb)

    cols = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
            'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
            'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

    #Limpieza de texto
    text = text.lower() #Pasar el texto a minúsculas
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Eliminar palabras cortas
    text = re.sub(r'\d+', '', text)  # Eliminar números
    #Exclusión de stopwords
    english_stopwords = nltk.corpus.stopwords.words('english')
    words = text.split()
    words = [word for word in words if word.lower() not in english_stopwords]
    #Stemming
    if stemming == True:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]  # Aplicar stemming
    #Lematización
    if lematize ==True:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words] 
    text_clean = ' '.join(words) #Texto limpio

    #Embeddings
    words = text_clean.split()
    embeddings = []
    for word in words:
        if word in embeddings_index: #Se debe contar con el indice de embeddings
            embeddings.append(embeddings_index[word])
    if len(embeddings) == 0:
        return np.zeros(embedding_dim)  # Si la palabra no existe se rellena del tamaño de la longitud de los embeddings
    
    text_embedded = np.mean(embeddings, axis=0) #Vector de embeddings del texto, promedia los valores
    text_embedded = text_embedded.reshape(1, embedding_dim)  #vector de embeddings

    #Predicción
    modelo = modelo_def
    prediccion =  modelo.predict(text_embedded)

    #Impresión
      
    dict_res = {}
    for i in range(0,len(cols)):
        dict_res[cols[i]] = prediccion[0][i].round(3)
    dict_res_Str = str(dict_res)

    return dict_res_Str


# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Movie Genre API',
    description='Movie genre prediction API')

ns = api.namespace('predict', 
     description='Movie genre Classifier')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument(
    'Texto', 
    type=str, 
    required=True, 
    help='Insert the description of the movie to classify', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

# Definición de la clase para disponibilización
@ns.route('/')
class MovieGenreApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": PrediceGenero(args['Texto'])
        }, 200
    
if __name__ == '__main__':        
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)